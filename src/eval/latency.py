"""Honest latency measurement: where the seconds go.

Two conditions, without which the number is a lie.

1. THE CACHE IS OFF. With the cache a repeat run answers in a millisecond, and a
   system that takes three seconds would report a hundredth of that. Debt #12 had
   been hanging since the very beginning.

2. WARM-UP. The first search call pulls bge-m3 off disk, the first rerank pulls
   another 568M-parameter model. Without a warm-up the first trace showed 10
   seconds for "dense search" over 427 vectors, which is physically impossible:
   that was weight loading. We measure the steady state, and report cold start
   separately - it is real too, just a different quantity.

What is measured. Self time by kind, so that the parts add up to the whole:
  llm        model calls
  retrieval  dense search
  rerank     cross-encoder
  tool       SQL and tool overhead
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
    """Warm up the models. Returns the cost of the cold start - useful on its
    own: that is what the first request after a deploy costs."""
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
    """Measuring the LOCAL stages: they need no provider quota and therefore
    reproduce for anyone at any time. They also account for the bulk of the
    budget, so there is no reason to postpone them until quota appears.
    """
    from src.tools.search import CANDIDATES, RERANK_WINDOW, get_search_tool
    from src.tools.sql import run as sql_run

    qs = ["How much does it cost to ship 2 kg to Bahia?",
          "What is the voluntary return window?",
          "When does Saturday count as a working day in Roraima?",
          "Which categories require prior authorisation for returns?",
          "What compensation is due for a delivery 5 days late?"]
    tool = get_search_tool()
    tool.search("warmup", top_k=3)                      # the warm-up is mandatory

    dense, rerank, sql = [], [], []
    traces: list[obs.Trace] = []
    for _ in range(n_runs):
        for qi, q in enumerate(qs):
            with obs.trace(f"local-{qi}", "stages") as tr:
                tool.search(q, top_k=5)
            traces.append(tr)
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
    print("LOCAL STAGES  (no provider quota, reproducible at any time)")
    print(f"{'stage':<26}{'n':>4}{'p50':>10}{'p95':>10}{'max':>10}")
    for name, vals in ((f"dense search, k={CANDIDATES}", dense),
                       (f"reranker, window {RERANK_WINDOW}", rerank),
                       ("SQL query", sql)):
        print(f"{name:<26}{len(vals):>4}{pct(vals,50):>8.0f}ms{pct(vals,95):>8.0f}ms"
              f"{max(vals):>8.0f}ms")
    # Traces are written here too: they let the Langfuse exporter be checked
    # without provider quota, on real search and rerank intervals.
    obs.dump(traces, EVALS / "traces.jsonl")
    tot = pct(dense, 50) + pct(rerank, 50)
    print()
    print(f"one search_docs call at the median: {tot:.0f} ms, "
          f"of which the reranker is {pct(rerank, 50) / tot * 100:.0f}%")
    print(f"one agent step = search_docs + one LLM call. The LLM contribution is "
          f"measured separately and needs provider quota.")


def main() -> None:
    from src.agent import loop, router
    from src.llm.providers import OpenAICompatProvider

    if os.getenv("LAT_LOCAL_ONLY") == "1":
        cold = warmup()
        print(f"cold start (loading the models): {cold / 1000:.1f} s")
        local_stages()
        return

    if os.getenv("LLM_CACHE") != "0":
        raise SystemExit("Run only with LLM_CACHE=0, otherwise the latency is fictional:\n"
                         "  LLM_CACHE=0 make latency")

    with GOLDEN_AGENT_PATH.open(encoding="utf-8") as f:
        golden = {json.loads(line)["id"]: json.loads(line) for line in f}
    qs = [golden[i] for i in SAMPLE]

    cold = warmup()
    print(f"cold start (loading the models): {cold / 1000:.1f} s")
    print(f"questions: {len(qs)}   runs: {RUNS}   model: {settings.llm_model}\n")

    llm = OpenAICompatProvider(settings.llm_provider, settings.llm_model)
    # The measurement has to survive a provider failure. The first version died
    # on a 429 and lost EVERYTHING it had computed - on a free tier that means
    # "the run is impossible in principle". Now every trace is written to disk
    # immediately, and a failure on one question does not take down the rest.
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
                    print(f"  {scheme:<7} run {r+1} {q['id']:<4} CRASH: "
                          f"{type(e).__name__}", flush=True)
                    continue
                p, c = tr.tokens()
                tr.outcome = {"kind": q["kind"], "run": r + 1,
                              "prompt_tokens": p, "completion_tokens": c}
                traces.append(tr)
                obs.dump(traces, EVALS / "traces.jsonl")     # incrementally
                print(f"  {scheme:<7} run {r+1} {q['id']:<4} "
                      f"{tr.total_ms/1000:>6.1f} s", flush=True)

    if not traces:
        raise SystemExit("no traces collected at all - check the provider quota")
    if failed:
        print(f"\nprovider failures: {failed}. The numbers below cover what was collected.")

    print(f"\n{'='*72}\nLATENCY BUDGET, SELF TIME BY STAGE")
    for scheme in ("router", "agent"):
        sel = [t for t in traces if t.scheme == scheme]
        tot = [t.total_ms for t in sel]
        print(f"\n--- {scheme}   n={len(sel)}   "
              f"p50 {pct(tot,50)/1000:.1f} s   p95 {pct(tot,95)/1000:.1f} s   "
              f"max {max(tot)/1000:.1f} s")
        agg = defaultdict(float)
        for t in sel:
            for k, v in t.by_kind().items():
                agg[k] += v
        s = sum(agg.values()) or 1.0
        print(f"    {'stage':<12}{'mean per request':>18}{'share':>8}")
        for k in KINDS:
            if agg.get(k):
                print(f"    {k:<12}{agg[k]/len(sel)/1000:>15.2f} s{agg[k]/s*100:>7.1f}%")
        pt = sum(t.outcome["prompt_tokens"] for t in sel) / len(sel)
        ct = sum(t.outcome["completion_tokens"] for t in sel) / len(sel)
        print(f"    tokens: input {pt:,.0f}   output {ct:,.0f}   "
              f"output share {ct/(pt+ct)*100:.1f}%")

    print(f"\nspread across questions (agent, p50 over {RUNS} runs):")
    for i in SAMPLE:
        v = [t.total_ms for t in traces if t.scheme == "agent" and t.label == i]
        print(f"    {i:<5}{golden[i]['kind']:<6}p50 {pct(v,50)/1000:>5.1f} s   "
              f"range {min(v)/1000:.1f}-{max(v)/1000:.1f} s")

    print(f"\ntraces: {EVALS/'traces.jsonl'}")


if __name__ == "__main__":
    main()