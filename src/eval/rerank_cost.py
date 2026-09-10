"""Reranker: quality against latency by window size.

Why a separate measurement. M2 (decision 11) established the QUALITY at windows
10/20/50. M4 (decision 32) established that the reranker eats 98% of search time.
What was missing is the pair: how much each window costs. Without it, K is chosen
by eye.

Computed locally, without a single provider call, and therefore always
reproducible. A warm-up is mandatory: the first call pulls a 568M-parameter model
off disk, and without it the measurement captures weight loading, not work.
"""
from __future__ import annotations

import json
import time

from src.config import EVALS
from src.eval.retrieval import load_golden

WINDOWS = (0, 5, 10, 20, 50)      # 0 = reranker off
REPEATS = 3

# recall@5 and recall@1 on the hard set, measured in M2 (evals/ablation_rerank.json).
# Not recomputed here: the point of this file is the PRICE; the quality is known.
QUALITY = {
    0:  {"r1": 0.765, "r5": 0.863},
    10: {"r1": 0.686, "r5": 0.902},
    20: {"r1": 0.667, "r5": 0.941},
    50: {"r1": 0.667, "r5": 0.961},
}


def pct(v: list[float], p: float) -> float:
    s = sorted(v)
    return s[min(int(round(p / 100 * (len(s) - 1))), len(s) - 1)]


def main() -> None:
    from src.rag.rerank import get_reranker
    from src.rag.retrieve import get_retriever

    qs = [g["question"] for g in load_golden(hard=True)][:12]
    r, rr = get_retriever(), get_reranker()
    r.search("warmup", k=50)
    rr.score("warmup", ["a"])                       # warm up both models

    rows = []
    for w in WINDOWS:
        lat = []
        for _ in range(REPEATS):
            for q in qs:
                t0 = time.perf_counter()
                hits = r.search(q, k=50 if w else 5)
                if w:
                    rr.score(q, [h.text for h in hits[:w]])
                lat.append((time.perf_counter() - t0) * 1000)
        rows.append({"window": w, "n": len(lat),
                     "p50": pct(lat, 50), "p95": pct(lat, 95), **QUALITY.get(w, {})})
        print(f"  window {w:>2}: p50 {pct(lat,50):>7.0f} ms   p95 {pct(lat,95):>7.0f} ms",
              flush=True)

    (EVALS / "rerank_cost.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                            encoding="utf-8")

    print("=" * 76)
    print("QUALITY AGAINST PRICE  (hard set, 51 questions)")
    hdr = (f"{'window':<7}{'recall@1':>10}{'recall@5':>10}{'p50 search':>13}"
           f"{'p95':>10}{'price @5':>17}")
    print(hdr); print("-" * len(hdr))
    base = rows[0]
    for x in rows:
        if "r5" not in x:
            print(f"{x['window']:<7}{'-':>10}{'-':>10}{x['p50']:>12.0f}ms{x['p95']:>9.0f}ms")
            continue
        d5 = x["r5"] - base["r5"]
        dms = x["p50"] - base["p50"]
        cost = f"{dms/(d5*100):.0f} ms/point" if d5 > 0 else "-"
        print(f"{x['window']:<7}{x['r1']:>10.3f}{x['r5']:>10.3f}"
              f"{x['p50']:>12.0f}ms{x['p95']:>9.0f}ms{cost:>17}")
    print()
    print("\"price @5\" is how many milliseconds one percentage point of recall@5 costs")
    print("relative to the reranker being off.")


if __name__ == "__main__":
    main()
