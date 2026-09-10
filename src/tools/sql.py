"""SQL tool: a question in words -> a query -> a result.

Three things make this an agent tool rather than a wrapper around duckdb.

1. ERRORS COME BACK AS TEXT, NOT AS EXCEPTIONS. The agent must get a chance to
   read the error message and fix the query. An exception kills the loop; the
   line "Binder Error: column X does not exist" teaches it.

2. THE QUERY IS VALIDATED BEFORE EXECUTION. The connection is opened read-only,
   so writes are impossible at the engine level. The extra check is not a
   replacement for that but exists for a comprehensible message: a refusal in a
   millisecond with an explanation beats an engine error a minute later.

3. ONLY THE TWO MARTS ARE VISIBLE, raw tables are hidden. The marts already
   encode the decision about grain; joining raw tables, the model would reproduce
   the fan-out the marts protect against. The restriction is deliberate: some
   questions (payments, sellers, geodata) become unanswerable. That is written
   into the backlog.

The schema description levels (SchemaLevel) exist for the sake of measurement:
the hypothesis that stating the grain explicitly lowers the share of grain errors
is checked with a number rather than taken on faith.
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache

import duckdb

from src.config import DB_PATH

VIEWS = ("order_facts", "order_summary")
MAX_ROWS = 50            # this many rows go into the model context
TIMEOUT_S = 20.0

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|install|"
    r"load|pragma|export|import|call|set|reset)\b", re.I)
FROM_JOIN = re.compile(r"\b(?:from|join)\s+([a-zA-Z_][\w.]*)", re.I)
CTE_NAMES = re.compile(r"\b(?:with|,)\s+([a-zA-Z_]\w*)\s+as\s*\(", re.I)


class SchemaLevel(IntEnum):
    """How much schema context the model gets. Each level is one ablation row."""
    NAMES = 1        # table and column names only
    GRAIN = 2        # + what one row is and where each fact lives
    SAMPLES = 3      # + sample rows
    FEWSHOT = 4      # + "question -> SQL" examples


@dataclass
class SqlResult:
    ok: bool
    sql: str
    columns: list[str]
    rows: list[tuple]
    error: str | None = None
    truncated: bool = False
    elapsed_ms: float = 0.0

    def as_text(self) -> str:
        """What the result looks like to the model."""
        if not self.ok:
            return f"SQL ERROR: {self.error}"
        if not self.rows:
            return "OK, 0 rows."
        head = " | ".join(self.columns)
        body = "\n".join(" | ".join(_fmt(v) for v in r) for r in self.rows)
        tail = f"\n... truncated at {MAX_ROWS} rows" if self.truncated else ""
        return f"{head}\n{body}{tail}"


def _fmt(v) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, float):
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return str(v)


@lru_cache(maxsize=1)
def _con() -> duckdb.DuckDBPyConnection:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"{DB_PATH} is missing. Run: make data")
    return duckdb.connect(str(DB_PATH), read_only=True)


def _tables_used(sql: str) -> set[str]:
    """Table names from the PARSED query, not from a regex over the text.

    A "from + word" regex breaks on EXTRACT(year FROM purchased_at): it takes
    `purchased_at` for a table and rejects a perfectly valid query. This defect
    threw away a correct model answer on the very first run - that is, it spoiled
    not the tool but the MEASUREMENT. We parse through DuckDB's own parser; the
    regex stays as a fallback.
    """
    try:
        raw = _con().execute("SELECT json_serialize_sql(?)", [sql]).fetchone()[0]
        out: set[str] = set()

        def walk(node):
            if isinstance(node, dict):
                if node.get("type") == "BASE_TABLE" and node.get("table_name"):
                    out.add(str(node["table_name"]).lower())
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)

        walk(json.loads(raw))
        return out
    except Exception:                       # noqa: BLE001 - the parser must not break the tool
        return {t.lower() for t in FROM_JOIN.findall(sql)}


def validate(sql: str) -> str | None:
    """The refusal reason, or None. Checked BEFORE execution, for a clear answer."""
    s = re.sub(r"--[^\n]*|/\*.*?\*/", " ", sql, flags=re.S).strip().rstrip(";")
    if not s:
        return "empty query"
    if ";" in s:
        return "only a single statement is allowed"
    if not re.match(r"^\s*(select|with)\b", s, re.I):
        return "only SELECT (or WITH ... SELECT) is allowed"
    if m := FORBIDDEN.search(s):
        return f"statement type '{m.group(1).upper()}' is not allowed, this tool is read-only"
    used = _tables_used(s)
    allowed = set(VIEWS) | {n.lower() for n in CTE_NAMES.findall(s)}
    if unknown := used - allowed:
        return (f"unknown or forbidden table(s): {', '.join(sorted(unknown))}. "
                f"Only these are available: {', '.join(VIEWS)}")
    return None


def run(sql: str) -> SqlResult:
    """Run the query. Never raises: an error is part of the answer."""
    import time

    if reason := validate(sql):
        return SqlResult(False, sql, [], [], error=reason)

    con, box = _con().cursor(), {}
    t0 = time.perf_counter()

    def work():
        try:
            cur = con.execute(sql)
            box["cols"] = [d[0] for d in cur.description]
            box["rows"] = cur.fetchmany(MAX_ROWS + 1)
        except Exception as e:                     # noqa: BLE001 - the error goes to the model as text
            box["err"] = f"{type(e).__name__}: {e}"

    th = threading.Thread(target=work, daemon=True)
    th.start()
    th.join(TIMEOUT_S)
    if th.is_alive():
        con.interrupt()
        th.join(2.0)
        return SqlResult(False, sql, [], [],
                         error=f"query exceeded the {TIMEOUT_S:.0f}s time limit")

    ms = (time.perf_counter() - t0) * 1000
    if "err" in box:
        return SqlResult(False, sql, [], [], error=box["err"], elapsed_ms=ms)

    rows, trunc = box["rows"], len(box["rows"]) > MAX_ROWS
    return SqlResult(True, sql, box["cols"], rows[:MAX_ROWS], truncated=trunc, elapsed_ms=ms)


# --- schema description: four levels ----------------------------------------
#
# The levels differ ONLY in the amount of context, never in how the task is
# phrased. Otherwise the comparison would measure the quality of the wording
# rather than the usefulness of the context.

GRAIN = {
    "order_facts": (
        "One row = ONE ITEM inside an order. "
        "An order with 3 items occupies 3 rows here, and every order-level "
        "column (status, dates, customer_state) is REPEATED in each of them."),
    "order_summary": (
        "One row = ONE ORDER, exactly once."),
}

FACT_HOME = (
    "Where each fact lives:\n"
    "  price, freight_value, product_id, seller_id, category -> item level, "
    "so they exist only in order_facts.\n"
    "  review_score, item_count, items_total, freight_total -> order level, "
    "so they exist only in order_summary.\n"
    "review_score is deliberately absent from order_facts: averaging it there "
    "would weight each order by its basket size and silently return a wrong number."
)

FEWSHOT = [
    ("How many delivered orders came from Sao Paulo state?",
     "SELECT COUNT(*) AS orders\nFROM order_summary\n"
     "WHERE customer_state = 'SP' AND order_status = 'delivered'"),
    ("What is the average item price in the bed_bath_table category?",
     "SELECT AVG(price) AS avg_price\nFROM order_facts\n"
     "WHERE category = 'bed_bath_table'"),
    ("Which three states have the lowest average review score?",
     "SELECT customer_state, AVG(review_score) AS avg_score\nFROM order_summary\n"
     "WHERE review_score IS NOT NULL\nGROUP BY customer_state\n"
     "ORDER BY avg_score ASC\nLIMIT 3"),
]


@lru_cache(maxsize=8)
def describe(level: SchemaLevel = SchemaLevel.GRAIN) -> str:
    con = _con()
    out = ["You may query ONLY these two read-only views. Raw tables are not available."]

    for v in VIEWS:
        cols = con.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = ? ORDER BY ordinal_position", [v]).fetchall()
        out.append(f"\n{v}")
        if level >= SchemaLevel.GRAIN:
            out.append(f"  {GRAIN[v]}")
        out.append("  " + ", ".join(f"{c} {t.lower()}" for c, t in cols))

        if level >= SchemaLevel.SAMPLES:
            names = [c for c, _ in cols][:6]
            rows = con.execute(
                f"SELECT {', '.join(names)} FROM {v} LIMIT 3").fetchall()
            out.append("  sample rows (" + ", ".join(names) + "):")
            out += [f"    {', '.join(_fmt(x) for x in r)}" for r in rows]

    if level >= SchemaLevel.GRAIN:
        out.append("\n" + FACT_HOME)
    if level >= SchemaLevel.FEWSHOT:
        out.append("\nExamples:")
        for q, s in FEWSHOT:
            out.append(f"  Q: {q}\n  SQL: {s.replace(chr(10), ' ')}")
    return "\n".join(out)


TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "sql_query",
        "description": (
            "Run a read-only SQL query against the e-commerce order database "
            "(DuckDB). Use it for counts, sums, averages, rankings and trends. "
            "Returns rows as text, or an error message you can act on."),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string",
                        "description": "A single SELECT statement. No semicolons."}},
            "required": ["sql"],
        },
    },
}


def main() -> None:
    """Manual check: show the schema levels and run a few queries."""
    print("=" * 70)
    print("LEVEL 1 (NAMES)\n")
    print(describe(SchemaLevel.NAMES))
    print("\n" + "=" * 70)
    print("LEVEL 2 (GRAIN), what it adds to the first\n")
    print(describe(SchemaLevel.GRAIN))

    checks = [
        ("valid query",     "SELECT AVG(review_score) AS s FROM order_summary"),
        ("grain error",     "SELECT AVG(review_score) AS s FROM order_facts"),
        ("write",           "DELETE FROM order_summary"),
        ("raw table",       "SELECT COUNT(*) FROM orders"),
        ("two statements",  "SELECT 1; DROP TABLE orders"),
        ("missing column",  "SELECT nope FROM order_summary"),
        ("CTE is allowed",  "WITH x AS (SELECT 1 AS a) SELECT a FROM x"),
    ]
    print("\n" + "=" * 70)
    print("TOOL BEHAVIOUR\n")
    for name, sql in checks:
        r = run(sql)
        status = "OK    " if r.ok else "REFUSE"
        print(f"[{status}] {name:<24} {r.as_text().splitlines()[0][:70]}")


if __name__ == "__main__":
    main()
