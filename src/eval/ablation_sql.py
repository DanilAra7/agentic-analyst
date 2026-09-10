"""Ablation: how the amount of schema description affects text-to-SQL quality.

The hypothesis checked with a number: stating the grain explicitly in the tool
description lowers the share of grain errors. Everyone writes this into their
prompt and nobody measures it.

Four description levels (SchemaLevel), the same set of 34 questions, the same
task wording. ONLY the amount of schema context changes.

How the outcome is classified. A plain "right/wrong" is not enough: we need to
know HOW the model went wrong, otherwise the gain is visible and its cause is not.
  CORRECT   the result matched the reference
  TRAP      the result matched the PREDICTED wrong query. This is the grain
            error: the model wrote valid SQL with the wrong meaning
  WRONG     wrong in some other way
  SQL_ERROR the query did not execute
  REFUSED   the model declined to answer
For unanswerable questions the scale flips: only REFUSED is correct, any number
is a failure. Otherwise the measurement would score a confident wrong answer.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict

from src.config import EVALS, settings
from src.eval.golden_sql import GOLDEN_SQL_PATH
from src.tools.sql import SchemaLevel, describe, run

REFUSAL = re.compile(r"\b(cannot|can't|not available|no such|unable|insufficient|"
                     r"not possible|does not exist|no data)\b", re.I)

PROMPT = """You are a SQL analyst. Write ONE DuckDB SELECT query that answers the question.

{schema}

Rules:
- Output the SQL query and nothing else. No explanation, no markdown fences.
- If the question cannot be answered from these views, output exactly: CANNOT ANSWER
- A single statement, no semicolon.

Question: {question}"""


REL_TOL = 1e-4   # 0.01%: absorbs a rounding difference, but not one of meaning


def as_num(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def same(a, b) -> bool:
    """Numbers are compared with a relative tolerance, not by their string form.

    The first run scored `AVG(item_count)` = 1.13277 as a failure against the
    reference 1.1328, because the reference was rounded inside the SQL itself
    while the model's answer was not. That is an INSTRUMENT error: we were
    measuring formatting instead of meaning. A 1e-4 threshold absorbs rounding
    without absorbing the traps, where the difference is in whole percent.
    """
    x, y = as_num(a), as_num(b)
    if x is not None and y is not None:
        if x == y:
            return True
        return abs(x - y) <= REL_TOL * max(abs(x), abs(y), 1e-12)
    if (a is None) != (b is None):
        return False
    return str(a).strip().lower() == str(b).strip().lower()


def rows_match(expected: list, actual: list) -> bool:
    """ROW order matters (it is the answer to a "top 3"), COLUMN order does not:
    the model is free to pick other names and another field order."""
    if len(expected) != len(actual):
        return False
    for e, a in zip(expected, actual):
        pool = list(a)
        for want in e:
            hit = next((i for i, got in enumerate(pool) if same(want, got)), None)
            if hit is None:
                return False
            pool.pop(hit)
    return True


def classify(q: dict, sql: str, res) -> str:
    if sql.strip().upper().startswith("CANNOT ANSWER") or (
            not sql.strip() and REFUSAL.search(sql)):
        return "REFUSED"
    if not res.ok:
        return "SQL_ERROR"
    actual = [list(r) for r in res.rows]
    if rows_match(q["expected"], actual):
        return "CORRECT"
    if q.get("wrong_expected") and rows_match(q["wrong_expected"], actual):
        return "TRAP"
    return "WRONG"


def is_good(q: dict, outcome: str) -> bool:
    """For unanswerable questions, only a refusal is correct."""
    return outcome == ("REFUSED" if q.get("unanswerable") else "CORRECT")


def main() -> None:
    from src.llm.providers import OpenAICompatProvider

    llm = OpenAICompatProvider(settings.llm_provider, settings.llm_model)
    with GOLDEN_SQL_PATH.open(encoding="utf-8") as f:
        golden = [json.loads(line) for line in f]
    print(f"questions: {len(golden)}   model: {llm.name}/{llm.model}\n")

    results, per_level = {}, {}
    for level in SchemaLevel:
        schema = describe(level)
        outcomes, by_trap, details = Counter(), defaultdict(Counter), []
        print(f"\n[{level.value} {level.name}] start", flush=True)
        for i, q in enumerate(golden, 1):
            t0 = time.perf_counter()
            r = llm.complete([{"role": "user",
                               "content": PROMPT.format(schema=schema, question=q["question"])}],
                             temperature=0.0, max_tokens=400)
            dt = time.perf_counter() - t0
            sql = re.sub(r"^```(?:sql)?|```$", "", r.text.strip(), flags=re.M).strip()
            res = run(sql)
            o = classify(q, sql, res)
            outcomes[o] += 1
            by_trap[q["trap"]][o] += 1
            details.append({"id": q["id"], "trap": q["trap"], "outcome": o,
                            "good": is_good(q, o), "sql": sql, "llm_s": round(dt, 1),
                            "cached": r.cached,
                            "error": res.error if not res.ok else None})
            # progress is printed for every question: a previous run stalled
            # silently, and finding out where took killing the process
            mark = "." if is_good(q, o) else "x"
            print(f"  {i:>2}/{len(golden)} {q['id']:<4}{mark} {o:<9} "
                  f"{dt:>5.1f}s{' (cached)' if r.cached else ''}", flush=True)
        good = sum(1 for d in details if d["good"])
        per_level[level.name] = {"good": good, "n": len(golden),
                                 "accuracy": good / len(golden),
                                 "outcomes": dict(outcomes),
                                 "by_trap": {k: dict(v) for k, v in by_trap.items()},
                                 "details": details}
        print(f"[{level.value} {level.name:<8}] accuracy {good/len(golden):.3f}   "
              + "  ".join(f"{k}={v}" for k, v in sorted(outcomes.items())), flush=True)

    (EVALS / "ablation_sql.json").write_text(
        json.dumps(per_level, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("GRAIN ERRORS (TRAP) BY LEVEL - the main question of the experiment")
    print(f"{'level':<12}{'accuracy':>10}{'TRAP':>7}{'WRONG':>7}{'SQL_ERROR':>11}{'REFUSED':>9}")
    for name, v in per_level.items():
        o = v["outcomes"]
        print(f"{name:<12}{v['accuracy']:>10.3f}{o.get('TRAP',0):>7}{o.get('WRONG',0):>7}"
              f"{o.get('SQL_ERROR',0):>11}{o.get('REFUSED',0):>9}")

    print("\nBY TRAP CLASS (share correct)")
    traps = sorted({t for v in per_level.values() for t in v["by_trap"]})
    print(f"{'class':<8}" + "".join(f"{n:<12}" for n in per_level))
    for t in traps:
        row = []
        for name, v in per_level.items():
            d = [x for x in v["details"] if x["trap"] == t]
            row.append(f"{sum(1 for x in d if x['good'])}/{len(d)}")
        print(f"{t:<8}" + "".join(f"{x:<12}" for x in row))
    print(f"\nwritten: {EVALS/'ablation_sql.json'}")


if __name__ == "__main__":
    main()
