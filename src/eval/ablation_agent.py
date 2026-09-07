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
        # Сохраняем ТРАССУ, а не только исход. Повод конкретный: вопрос bq5 в
        # одном прогоне из трёх дал 6 449 вместо 7 073, и восстановить, каким
        # запросом получено это число, оказалось невозможно - в файле лежали
        # только метрики. Разбор провала требует шагов с аргументами.
        rows.append({"id": q["id"], "kind": q["kind"], "question": q["question"],
                     "answer": tr.answer,
                     "trace": [{"tool": st.tool, "args": st.arguments,
                                "result": st.result[:400], "ok": st.ok} for st in tr.steps],
                     **g})
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


def summarise(runs: list[dict]) -> None:
    """Среднее и РАЗМАХ по нескольким прогонам.

    Один прогон - не измерение (бэклог №37): два прогона агента до этого дали
    0.88 и 1.00 на двухисточниковых, разница в один вопрос равна 0.125 при n=8.
    Без размаха любое сравнение конфигураций рискует обсуждать шум.
    Прогоны обязаны идти с LLM_CACHE=0, иначе повторы вернут тот же ответ и
    размах окажется нулевым по построению.
    """
    print(f"\n=== {len(runs)} ПРОГОНА: среднее и размах ===")
    hdr = f"{'тип':<7}{'n':>3}{'верно':>22}{'шагов':>16}{'токенов':>18}"
    print(hdr); print("-" * len(hdr))

    def cell(vals, fmt="{:.2f}"):
        lo, hi = min(vals), max(vals)
        mid = sum(vals) / len(vals)
        span = "" if lo == hi else f" [{fmt.format(lo)}-{fmt.format(hi)}]"
        return fmt.format(mid) + span

    for k in KINDS + ("ВСЕГО",):
        per = []
        for r in runs:
            rows = r["rows"] if k == "ВСЕГО" else [x for x in r["rows"] if x["kind"] == k]
            if rows:
                per.append(rows)
        if not per:
            continue
        n = len(per[0])
        print(f"{k:<7}{n:>3}"
              f"{cell([sum(x['correct'] for x in g)/len(g) for g in per]):>22}"
              f"{cell([sum(x['steps'] for x in g)/len(g) for g in per], '{:.1f}'):>16}"
              f"{cell([sum(x['tokens'] for x in g)/len(g) for g in per], '{:.0f}'):>18}")

    # какие вопросы вели себя нестабильно - это интереснее среднего
    ids = {x["id"] for x in runs[0]["rows"]}
    flaky = [i for i in sorted(ids)
             if len({tuple(x["correct"] for x in r["rows"] if x["id"] == i) for r in runs}) > 1]
    print(f"\nнестабильные вопросы: {', '.join(flaky) if flaky else 'нет'}")


def main() -> None:
    import os

    from src.agent import loop, router
    from src.llm.providers import OpenAICompatProvider

    which = sys.argv[1] if len(sys.argv) > 1 else "router"
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    fn = {"router": router.answer, "agent": loop.answer}
    if which not in fn:
        raise SystemExit(f"неизвестная схема {which!r}, доступны: {list(fn)}")

    llm = OpenAICompatProvider(settings.llm_provider, settings.llm_model)
    golden = load()
    cache = "выключен" if os.getenv("LLM_CACHE") == "0" else "ВКЛЮЧЁН"
    print(f"схема: {which}   вопросов: {len(golden)}   прогонов: {n_runs}   "
          f"модель: {llm.name}/{llm.model}   кеш: {cache}")
    if n_runs > 1 and cache != "выключен":
        print("  ВНИМАНИЕ: с включённым кешем повторы вернут тот же ответ, "
              "размах будет нулевым по построению")
    print()

    runs = []
    for k in range(n_runs):
        if n_runs > 1:
            print(f"--- прогон {k + 1}/{n_runs}")
        runs.append(run_scheme(which, fn[which], llm, golden))

    report(runs[-1])
    if n_runs > 1:
        summarise(runs)

    path = EVALS / f"agent_{which}.json"
    path.write_text(json.dumps({"scheme": which, "runs": runs}, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"\nзаписано: {path}")


if __name__ == "__main__":
    main()
