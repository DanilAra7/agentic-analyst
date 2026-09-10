"""Running one scheme (router or agent) over the set of 24 questions.

Breaking the numbers down by question type is mandatory, not nice to have: the
router and the agent differ EXACTLY on the two-source questions. Overall accuracy
would average that difference away and hide it - the same mistake we already
caught in M2 with the reranker.

The criterion was fixed in advance, decision #18:
  sql / docs   the router should keep up with the agent
  both         the router should fail
  none         both schemes should refuse
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
            print(f"  {i:>2}/{len(golden)} {q['id']} CRASH: {type(e).__name__}: {e}", flush=True)
            continue
        g = grade(q, tr)
        # We store the TRACE, not just the outcome. The reason is concrete:
        # question bq5 in one run out of three returned 6,449 instead of 7,073,
        # and reconstructing which query produced that number turned out to be
        # impossible - the file held only metrics. Debugging a failure needs the
        # steps together with their arguments.
        rows.append({"id": q["id"], "kind": q["kind"], "question": q["question"],
                     "answer": tr.answer,
                     "trace": [{"tool": st.tool, "args": st.arguments,
                                "result": st.result[:400], "ok": st.ok} for st in tr.steps],
                     **g})
        mark = "+" if g["correct"] else "."
        print(f"  {i:>2}/{len(golden)} {q['id']:<4} {q['kind']:<5} {mark} "
              f"{','.join(g['tools_used']) or '-':<24}"
              f"{'tools ok' if g['tools_ok'] else 'TOOLS WRONG':<13}"
              f"{g['stop_reason']:<21}{time.perf_counter()-t0:>5.1f}s", flush=True)
    return {"scheme": name, "rows": rows}


def report(res: dict) -> None:
    rows = res["rows"]
    by = defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)

    print(f"\n=== {res['scheme'].upper()} ===")
    hdr = (f"{'kind':<7}{'n':>3}{'correct':>8}{'tools':>10}{'steps':>8}"
           f"{'calls':>9}{'tokens':>10}{'2nd hop':>11}")
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
    line("TOTAL", rows)

    bad = [r for r in rows if not r["correct"]]
    print(f"\nfailures ({len(bad)}):")
    for r in bad:
        print(f"  [{r['id']} {r['kind']}] {r['question'][:64]}")
        print(f"      tools={','.join(r['tools_used']) or '-'}   {r['stop_reason']}")
        print(f"      answer: {(r['answer'] or '').strip()[:150]}")


def summarise(runs: list[dict]) -> None:
    """Mean and SPREAD across several runs.

    One run is not a measurement (backlog #37): two earlier agent runs gave 0.88
    and 1.00 on the two-source questions, and a difference of one question equals
    0.125 at n=8. Without the spread, any comparison of configurations risks
    discussing noise.
    The runs must go with LLM_CACHE=0, otherwise repeats return the same answer
    and the spread is zero by construction.
    """
    print(f"\n=== {len(runs)} RUNS: mean and spread ===")
    hdr = f"{'kind':<7}{'n':>3}{'correct':>22}{'steps':>16}{'tokens':>18}"
    print(hdr); print("-" * len(hdr))

    def cell(vals, fmt="{:.2f}"):
        lo, hi = min(vals), max(vals)
        mid = sum(vals) / len(vals)
        span = "" if lo == hi else f" [{fmt.format(lo)}-{fmt.format(hi)}]"
        return fmt.format(mid) + span

    for k in KINDS + ("TOTAL",):
        per = []
        for r in runs:
            rows = r["rows"] if k == "TOTAL" else [x for x in r["rows"] if x["kind"] == k]
            if rows:
                per.append(rows)
        if not per:
            continue
        n = len(per[0])
        print(f"{k:<7}{n:>3}"
              f"{cell([sum(x['correct'] for x in g)/len(g) for g in per]):>22}"
              f"{cell([sum(x['steps'] for x in g)/len(g) for g in per], '{:.1f}'):>16}"
              f"{cell([sum(x['tokens'] for x in g)/len(g) for g in per], '{:.0f}'):>18}")

    # which questions behaved unstably - more interesting than the mean
    ids = {x["id"] for x in runs[0]["rows"]}
    flaky = [i for i in sorted(ids)
             if len({tuple(x["correct"] for x in r["rows"] if x["id"] == i) for r in runs}) > 1]
    print(f"\nunstable questions: {', '.join(flaky) if flaky else 'none'}")


def main() -> None:
    import os

    from src.agent import loop, router
    from src.llm.providers import OpenAICompatProvider

    which = sys.argv[1] if len(sys.argv) > 1 else "router"
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    fn = {"router": router.answer, "agent": loop.answer}
    if which not in fn:
        raise SystemExit(f"unknown scheme {which!r}, available: {list(fn)}")

    llm = OpenAICompatProvider(settings.llm_provider, settings.llm_model)
    golden = load()
    cache = "off" if os.getenv("LLM_CACHE") == "0" else "ON"
    print(f"scheme: {which}   questions: {len(golden)}   runs: {n_runs}   "
          f"model: {llm.name}/{llm.model}   cache: {cache}")
    if n_runs > 1 and cache != "off":
        print("  WARNING: with the cache on, repeats return the same answer and "
              "the spread will be zero by construction")
    print()

    runs = []
    for k in range(n_runs):
        if n_runs > 1:
            print(f"--- run {k + 1}/{n_runs}")
        runs.append(run_scheme(which, fn[which], llm, golden))

    report(runs[-1])
    if n_runs > 1:
        summarise(runs)

    path = EVALS / f"agent_{which}.json"
    path.write_text(json.dumps({"scheme": which, "runs": runs}, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
