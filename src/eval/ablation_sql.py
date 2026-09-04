"""Ablation: как объём описания схемы влияет на качество text-to-SQL.

Гипотеза, которую проверяем числом: явное указание зерна в описании инструмента
снижает долю ошибок зерна. Все так пишут в промпте и никто не меряет.

Четыре уровня описания (SchemaLevel), один и тот же набор из 34 вопросов,
одна и та же формулировка задачи. Меняется ТОЛЬКО объём контекста о схеме.

Как классифицируется исход. Просто «верно/неверно» мало: надо знать, ЧЕМ
именно ошиблась модель, иначе прирост будет виден, а его причина - нет.
  CORRECT   результат совпал с эталоном
  TRAP      результат совпал с ПРЕДСКАЗАННЫМ неверным запросом. Это и есть
            ошибка зерна: модель написала валидный SQL с неверным смыслом
  WRONG     ошиблась как-то иначе
  SQL_ERROR запрос не выполнился
  REFUSED   модель отказалась отвечать
Для неотвечаемых вопросов шкала переворачивается: верен только REFUSED,
любое число - провал. Иначе замер засчитал бы уверенный неверный ответ.
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


REL_TOL = 1e-4   # 0.01%: гасит разницу округления, но не разницу смысла


def as_num(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def same(a, b) -> bool:
    """Числа сравниваем с относительным допуском, а не по строковому виду.

    Первый прогон засчитал провалом `AVG(item_count)` = 1.13277 против эталона
    1.1328, потому что эталон был округлён в самом SQL, а ответ модели нет.
    Это ошибка ПРИБОРА: мы мерили форматирование вместо смысла. Порог 1e-4
    гасит округление и при этом не гасит ловушки: там разница в проценты.
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
    """Порядок СТРОК важен (это ответ на «топ-3»), порядок КОЛОНОК - нет:
    модель вправе выбрать другие имена и другой порядок полей."""
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
    """Для неотвечаемых вопросов верен только отказ."""
    return outcome == ("REFUSED" if q.get("unanswerable") else "CORRECT")


def main() -> None:
    from src.llm.providers import OpenAICompatProvider

    llm = OpenAICompatProvider(settings.llm_provider, settings.llm_model)
    with GOLDEN_SQL_PATH.open(encoding="utf-8") as f:
        golden = [json.loads(line) for line in f]
    print(f"вопросов: {len(golden)}   модель: {llm.name}/{llm.model}\n")

    results, per_level = {}, {}
    for level in SchemaLevel:
        schema = describe(level)
        outcomes, by_trap, details = Counter(), defaultdict(Counter), []
        print(f"\n[{level.value} {level.name}] старт", flush=True)
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
            # прогресс печатаем на каждом вопросе: прошлый прогон встал молча,
            # и понять, на чём именно, удалось только убив процесс
            mark = "." if is_good(q, o) else "x"
            print(f"  {i:>2}/{len(golden)} {q['id']:<4}{mark} {o:<9} "
                  f"{dt:>5.1f}s{' (кеш)' if r.cached else ''}", flush=True)
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
    print("ОШИБКИ ЗЕРНА (TRAP) ПО УРОВНЯМ — главный вопрос эксперимента")
    print(f"{'уровень':<12}{'accuracy':>10}{'TRAP':>7}{'WRONG':>7}{'SQL_ERROR':>11}{'REFUSED':>9}")
    for name, v in per_level.items():
        o = v["outcomes"]
        print(f"{name:<12}{v['accuracy']:>10.3f}{o.get('TRAP',0):>7}{o.get('WRONG',0):>7}"
              f"{o.get('SQL_ERROR',0):>11}{o.get('REFUSED',0):>9}")

    print("\nПО КЛАССАМ ЛОВУШЕК (доля верных)")
    traps = sorted({t for v in per_level.values() for t in v["by_trap"]})
    print(f"{'класс':<8}" + "".join(f"{n:<12}" for n in per_level))
    for t in traps:
        row = []
        for name, v in per_level.items():
            d = [x for x in v["details"] if x["trap"] == t]
            row.append(f"{sum(1 for x in d if x['good'])}/{len(d)}")
        print(f"{t:<8}" + "".join(f"{x:<12}" for x in row))
    print(f"\nзаписано: {EVALS/'ablation_sql.json'}")


if __name__ == "__main__":
    main()
