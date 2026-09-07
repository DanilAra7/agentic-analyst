"""Честный замер латентности: куда уходят секунды.

Два условия, без которых цифра будет ложью.

1. КЕШ ВЫКЛЮЧЕН. С кешем повторный прогон отвечает за миллисекунду, и система,
   отвечающая три секунды, покажет сотую долю. Долг №12 висел с самого начала.

2. ПРОГРЕВ. Первый вызов поиска тянет с диска bge-m3, первый реранк - ещё одну
   модель на 568M параметров. Без прогрева первая трасса показала 10 секунд на
   «плотный поиск» по 427 векторам, чего физически быть не может: это была
   загрузка весов. Меряем установившийся режим, а холодный старт называем
   отдельно - он тоже реальность, просто другая величина.

Что меряем. Собственное время по видам, чтобы части складывались в целое:
  llm        вызовы модели
  retrieval  плотный поиск
  rerank     cross-encoder
  tool       SQL и накладные расходы инструментов
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict

from src.config import EVALS, settings
from src.eval.golden_agent import GOLDEN_AGENT_PATH
from src.obs import trace as obs

SAMPLE = tuple(os.getenv("LAT_SAMPLE", "sq1,dq4,bq1,bq4,nq2").split(","))
RUNS = int(os.getenv("LAT_RUNS", "3"))
KINDS = ("llm", "retrieval", "rerank", "tool")


def warmup() -> float:
    """Прогреть модели. Возвращает стоимость холодного старта - она полезна
    сама по себе: столько стоит первый запрос после деплоя."""
    from src.tools.search import get_search_tool

    t0 = time.perf_counter()
    get_search_tool().search("warmup query about shipping deadlines", top_k=3)
    return (time.perf_counter() - t0) * 1000


def pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    i = min(int(round(p / 100 * (len(s) - 1))), len(s) - 1)
    return s[i]


def local_stages(n_runs: int = 5) -> None:
    """Замер ЛОКАЛЬНЫХ этапов: они не требуют квоты провайдера и потому
    воспроизводятся у кого угодно в любой момент. На них же приходится
    основная часть бюджета, так что откладывать их до появления квоты незачем.
    """
    from src.tools.search import CANDIDATES, RERANK_WINDOW, get_search_tool
    from src.tools.sql import run as sql_run

    qs = ["How much does it cost to ship 2 kg to Bahia?",
          "What is the voluntary return window?",
          "When does Saturday count as a working day in Roraima?",
          "Which categories require prior authorisation for returns?",
          "What compensation is due for a delivery 5 days late?"]
    tool = get_search_tool()
    tool.search("warmup", top_k=3)                      # прогрев обязателен

    dense, rerank, sql = [], [], []
    for _ in range(n_runs):
        for q in qs:
            with obs.trace("local", "stages") as tr:
                tool.search(q, top_k=5)
            for sp, own in zip(tr.spans, tr.self_ms()):
                (dense if sp.kind == "retrieval" else rerank).append(own)
    for _ in range(n_runs):
        for q in ("SELECT COUNT(*) FROM order_summary",
                  "SELECT customer_state, COUNT(*) FROM order_summary GROUP BY 1 "
                  "ORDER BY 2 DESC LIMIT 5",
                  "SELECT AVG(price) FROM order_facts WHERE category='bed_bath_table'"):
            t0 = time.perf_counter()
            sql_run(q)
            sql.append((time.perf_counter() - t0) * 1000)

    print("=" * 72)
    print("ЛОКАЛЬНЫЕ ЭТАПЫ  (без квоты провайдера, воспроизводятся всегда)")
    print(f"{'этап':<26}{'n':>4}{'p50':>10}{'p95':>10}{'макс':>10}")
    for name, vals in ((f"плотный поиск, k={CANDIDATES}", dense),
                       (f"реранкер, окно {RERANK_WINDOW}", rerank),
                       ("SQL-запрос", sql)):
        print(f"{name:<26}{len(vals):>4}{pct(vals,50):>9.0f}м{pct(vals,95):>9.0f}м"
              f"{max(vals):>9.0f}м")
    tot = pct(dense, 50) + pct(rerank, 50)
    print()
    print(f"один вызов search_docs по медиане: {tot:.0f} мс, "
          f"из них реранкер {pct(rerank, 50) / tot * 100:.0f}%")
    print(f"один шаг агента = search_docs + вызов LLM. Вклад LLM меряется "
          f"отдельно и требует квоты провайдера.")


def main() -> None:
    from src.agent import loop, router
    from src.llm.providers import OpenAICompatProvider

    if os.getenv("LAT_LOCAL_ONLY") == "1":
        cold = warmup()
        print(f"холодный старт (загрузка моделей): {cold / 1000:.1f} с")
        local_stages()
        return

    if os.getenv("LLM_CACHE") != "0":
        raise SystemExit("Запускать только с LLM_CACHE=0, иначе латентность фиктивна:\n"
                         "  LLM_CACHE=0 make latency")

    with GOLDEN_AGENT_PATH.open(encoding="utf-8") as f:
        golden = {json.loads(line)["id"]: json.loads(line) for line in f}
    qs = [golden[i] for i in SAMPLE]

    cold = warmup()
    print(f"холодный старт (загрузка моделей): {cold / 1000:.1f} с")
    print(f"вопросов: {len(qs)}   прогонов: {RUNS}   модель: {settings.llm_model}\n")

    llm = OpenAICompatProvider(settings.llm_provider, settings.llm_model)
    # Замер обязан переживать сбой провайдера. Первая версия падала на 429
    # и теряла ВСЁ, что успела посчитать, - при бесплатном тарифе это значит
    # «прогон невозможен в принципе». Теперь каждая трасса пишется на диск
    # сразу, а сбой на одном вопросе не роняет остальные.
    traces: list[obs.Trace] = []
    failed = 0
    for scheme, fn in (("router", router.answer), ("agent", loop.answer)):
        for r in range(RUNS):
            for q in qs:
                try:
                    with obs.trace(q["id"], scheme) as tr:
                        fn(llm, q["question"])
                except Exception as e:                       # noqa: BLE001
                    failed += 1
                    print(f"  {scheme:<7} прогон {r+1} {q['id']:<4} СБОЙ: "
                          f"{type(e).__name__}", flush=True)
                    continue
                p, c = tr.tokens()
                tr.outcome = {"kind": q["kind"], "run": r + 1,
                              "prompt_tokens": p, "completion_tokens": c}
                traces.append(tr)
                obs.dump(traces, EVALS / "traces.jsonl")     # инкрементально
                print(f"  {scheme:<7} прогон {r+1} {q['id']:<4} "
                      f"{tr.total_ms/1000:>6.1f} с", flush=True)

    if not traces:
        raise SystemExit("ни одной трассы не собрано - проверь квоту провайдера")
    if failed:
        print(f"\nсбоев провайдера: {failed}. Числа ниже - по тому, что собралось.")

    print(f"\n{'='*72}\nБЮДЖЕТ ЛАТЕНТНОСТИ, СОБСТВЕННОЕ ВРЕМЯ ПО ЭТАПАМ")
    for scheme in ("router", "agent"):
        sel = [t for t in traces if t.scheme == scheme]
        tot = [t.total_ms for t in sel]
        print(f"\n--- {scheme}   n={len(sel)}   "
              f"p50 {pct(tot,50)/1000:.1f} с   p95 {pct(tot,95)/1000:.1f} с   "
              f"макс {max(tot)/1000:.1f} с")
        agg = defaultdict(float)
        for t in sel:
            for k, v in t.by_kind().items():
                agg[k] += v
        s = sum(agg.values()) or 1.0
        print(f"    {'этап':<12}{'сред. на запрос':>18}{'доля':>8}")
        for k in KINDS:
            if agg.get(k):
                print(f"    {k:<12}{agg[k]/len(sel)/1000:>15.2f} с{agg[k]/s*100:>7.1f}%")
        pt = sum(t.outcome["prompt_tokens"] for t in sel) / len(sel)
        ct = sum(t.outcome["completion_tokens"] for t in sel) / len(sel)
        print(f"    токенов: вход {pt:,.0f}   выход {ct:,.0f}   "
              f"доля выхода {ct/(pt+ct)*100:.1f}%")

    print(f"\nразброс по вопросам (агент, p50 из {RUNS} прогонов):")
    for i in SAMPLE:
        v = [t.total_ms for t in traces if t.scheme == "agent" and t.label == i]
        print(f"    {i:<5}{golden[i]['kind']:<6}p50 {pct(v,50)/1000:>5.1f} с   "
              f"размах {min(v)/1000:.1f}-{max(v)/1000:.1f} с")

    print(f"\nтрассы: {EVALS/'traces.jsonl'}")


if __name__ == "__main__":
    main()