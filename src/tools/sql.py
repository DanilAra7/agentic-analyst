"""Инструмент SQL: вопрос словами -> запрос -> результат.

Три вещи, которые делают это инструментом агента, а не обёрткой над duckdb.

1. ОШИБКИ ВОЗВРАЩАЮТСЯ ТЕКСТОМ, А НЕ ИСКЛЮЧЕНИЕМ. Агент должен иметь шанс
   прочитать сообщение об ошибке и исправить запрос. Исключение убивает цикл,
   строка «Binder Error: column X does not exist» его учит.

2. ЗАПРОС ПРОВЕРЯЕТСЯ ДО ВЫПОЛНЕНИЯ. Соединение открыто read-only, так что
   запись невозможна на уровне движка. Дополнительная проверка нужна не вместо
   этого, а ради понятного сообщения: отказ за миллисекунду с объяснением
   полезнее, чем ошибка движка через минуту.

3. ВИДНЫ ТОЛЬКО ДВЕ ВИТРИНЫ, сырые таблицы скрыты. Витрины уже содержат
   решение про зерно; соединяя сырые таблицы, модель воспроизведёт fan-out,
   от которого витрины и защищают. Ограничение сознательное: часть вопросов
   (оплаты, продавцы, геоданные) станет неотвечаемой. Это записано в бэклог.

Уровни описания схемы (SchemaLevel) существуют ради замера: гипотеза, что
явное указание зерна снижает долю ошибок зерна, проверяется числом, а не
принимается на веру.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache

import duckdb

from src.config import DB_PATH

VIEWS = ("order_facts", "order_summary")
MAX_ROWS = 50            # столько строк уходит в контекст модели
TIMEOUT_S = 20.0

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|install|"
    r"load|pragma|export|import|call|set|reset)\b", re.I)
FROM_JOIN = re.compile(r"\b(?:from|join)\s+([a-zA-Z_][\w.]*)", re.I)
CTE_NAMES = re.compile(r"\b(?:with|,)\s+([a-zA-Z_]\w*)\s+as\s*\(", re.I)


class SchemaLevel(IntEnum):
    """Сколько контекста о схеме получает модель. Каждый уровень - строка ablation."""
    NAMES = 1        # только имена таблиц и колонок
    GRAIN = 2        # + чем является одна строка и где живёт какой факт
    SAMPLES = 3      # + примеры строк
    FEWSHOT = 4      # + примеры «вопрос -> SQL»


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
        """Как результат выглядит для модели."""
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
        raise FileNotFoundError(f"Нет {DB_PATH}. Запусти: make data")
    return duckdb.connect(str(DB_PATH), read_only=True)


def validate(sql: str) -> str | None:
    """Причина отказа или None. Проверяем ДО выполнения, ради внятного ответа."""
    s = re.sub(r"--[^\n]*|/\*.*?\*/", " ", sql, flags=re.S).strip().rstrip(";")
    if not s:
        return "empty query"
    if ";" in s:
        return "only a single statement is allowed"
    if not re.match(r"^\s*(select|with)\b", s, re.I):
        return "only SELECT (or WITH ... SELECT) is allowed"
    if m := FORBIDDEN.search(s):
        return f"statement type '{m.group(1).upper()}' is not allowed, this tool is read-only"
    allowed = set(VIEWS) | {n.lower() for n in CTE_NAMES.findall(s)}
    used = {t.lower() for t in FROM_JOIN.findall(s)}
    if unknown := used - allowed:
        return (f"unknown or forbidden table(s): {', '.join(sorted(unknown))}. "
                f"Only these are available: {', '.join(VIEWS)}")
    return None


def run(sql: str) -> SqlResult:
    """Выполнить запрос. Никогда не бросает исключение: ошибка - часть ответа."""
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
        except Exception as e:                     # noqa: BLE001 - ошибка идёт модели текстом
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


# --- описание схемы: четыре уровня ------------------------------------------
#
# Уровни отличаются ТОЛЬКО объёмом контекста, не формулировками задачи.
# Иначе сравнение измеряло бы качество формулировки, а не пользу контекста.

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
    """Ручная проверка: показать уровни схемы и прогнать несколько запросов."""
    print("=" * 70)
    print("УРОВЕНЬ 1 (NAMES)\n")
    print(describe(SchemaLevel.NAMES))
    print("\n" + "=" * 70)
    print("УРОВЕНЬ 2 (GRAIN), добавка к первому\n")
    print(describe(SchemaLevel.GRAIN))

    checks = [
        ("верный запрос", "SELECT AVG(review_score) AS s FROM order_summary"),
        ("ошибка зерна", "SELECT AVG(review_score) AS s FROM order_facts"),
        ("запись",       "DELETE FROM order_summary"),
        ("сырая таблица", "SELECT COUNT(*) FROM orders"),
        ("две команды",  "SELECT 1; DROP TABLE orders"),
        ("несуществующая колонка", "SELECT nope FROM order_summary"),
        ("CTE разрешён", "WITH x AS (SELECT 1 AS a) SELECT a FROM x"),
    ]
    print("\n" + "=" * 70)
    print("ПОВЕДЕНИЕ ИНСТРУМЕНТА\n")
    for name, sql in checks:
        r = run(sql)
        status = "OK " if r.ok else "ОТКАЗ"
        print(f"[{status}] {name:<24} {r.as_text().splitlines()[0][:70]}")


if __name__ == "__main__":
    main()
