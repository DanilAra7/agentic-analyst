"""Прогон схемы (роутер или агент) по набору из 24 вопросов.

Разбивка по типам вопросов обязательна, а не желательна: роутер и агент
отличаются РОВНО на двухисточниковых. Общая точность усреднит эту разницу и
скроет её - та же ошибка, что мы уже ловили в M2 с реранкером.

Критерий назначен заранее, решение №18:
  sql / docs   роутер должен идти наравне с агентом
  both         роутер должен провалиться
  none         обе схемы должны отказаться
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict

from src.agent.base import grade
from src.config import EVALS, settings
from src.eval.golden_agent import GOLDEN_AGENT_PATH

KINDS = ("sql", "docs", "both", "none")


def load() -> list[dict]:
    with GOLDEN_AGENT_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run_scheme(name: str, answer_fn, llm, golden: list[dict]) -> dict:
    rows = []
    for i, q in enumerate(golden, 1):
        t0 = time.perf_counter()
        try:
            tr = answer_fn(llm, q["question"])
        except Exception as e:                       # noqa: BLE001
            print(f"  {i:>2}/{len(golden)} {q['id']} СБОЙ: {type(e).__name__}: {e}", flush=True)
            continue
        g = grade(q, tr)
        rows.append({"id": q["id"], "kind": q["kind"], "question": q["question"],
                     "answer": tr.answer, **g})
        mark = "+" if g["correct"] else "."
        print(f"  {i:>2}/{len(golden)} {q['id']:<4} {q['kind']:<5} {mark} "
              f"{','.join(g['tools_used']) or '—':<24}"
              f"{'инстр.верно' if g['tools_ok'] else 'ИНСТР.МИМО':<13}"
              f"{g['stop_reason']:<21}{time.perf_counter()-t0:>5.1f}s", flush=True)
    return {"scheme": name, "rows": rows}


def report(res: dict) -> None:
    rows = res["rows"]
    by = defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)

    print(f"\n=== {res['scheme'].upper()} ===")
    hdr = (f"{'тип':<7}{'n':>3}{'верно':>8}{'инструм.':>10}{'шагов':>8}"
           f"{'вызовов':>9}{'токенов':>10}{'2-й раунд':>11}")
    print(hdr); print("-" * len(hdr))

    def line(label, g):
        n = len(g)
        print(f"{label:<7}{n:>3}{sum(r['correct'] for r in g)/n:>8.2f}"
              f"{sum(r['tools_ok'] for r in g)/n:>10.2f}"
              f"{sum(r['steps'] for r in g)/n:>8.1f}"
              f"{sum(r['llm_calls'] for r in g)/n:>9.1f}"
              f"{sum(r['tokens'] for r in g)/n:>10.0f}"
              f"{sum(r['stop_reason']=='needed_second_hop' for r in g)/n:>11.2f}")

    for k in KINDS:
        if by.get(k):
            line(k, by[k])
    print("-" * len(hdr))
    line("ВСЕГО", rows)

    bad = [r for r in rows if not r["correct"]]
    print(f"\nпровалы ({len(bad)}):")
    for r in bad:
        print(f"  [{r['id']} {r['kind']}] {r['question'][:64]}")
        print(f"      инстр={','.join(r['tools_used']) or '—'}   {r['stop_reason']}")
        print(f"      ответ: {(r['answer'] or '').strip()[:150]}")


def main() -> None:
    from src.agent import router
    from src.llm.providers import OpenAICompatProvider

    which = sys.argv[1] if len(sys.argv) > 1 else "router"
    fn = {"router": router.answer}
    if which not in fn:
        raise SystemExit(f"неизвестная схема {which!r}, доступны: {list(fn)}")

    llm = OpenAICompatProvider(settings.llm_provider, settings.llm_model)
    golden = load()
    print(f"схема: {which}   вопросов: {len(golden)}   модель: {llm.name}/{llm.model}\n")
    res = run_scheme(which, fn[which], llm, golden)
    report(res)

    path = EVALS / f"agent_{which}.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nзаписано: {path}")


if __name__ == "__main__":
    main()
