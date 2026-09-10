"""Conditional reranking: skip the reranker when dense search is already confident.

The idea comes from backlog #47(d), and it aims at two problems at once, not one.

  LATENCY  reranking costs 2.1 s at window 20 against 26 ms without it.
           Skipping it where it is not needed drops the average price.
  QUALITY  decision #13: the reranker DROPS recall@1 (0.765 -> 0.667), because it
           prefers a topically coherent document over a table row. Skipping it on
           confident queries wins part of that loss back.

The signal is the CONFIDENCE GAP: the cosine difference between the first and the
second candidate of dense search. A large gap means search tells the leader apart;
a small one means the candidates are bunched together and their order is arbitrary
- that is where the reranker is needed.

The threshold is NOT eyeballed: we sweep it and watch what happens to both metrics
and to the share of reranked queries.

Runs locally. The reranker logits are computed ONCE for all questions, after which
sweeping thresholds is free.
"""
from __future__ import annotations

import json

import numpy as np

from src.config import EVALS
from src.eval.retrieval import load_golden

WINDOW = 20
DENSE_MS, RERANK_MS = 26.0, 2100.0     # measured in src/eval/rerank_cost.py
THRESHOLDS = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 1.0)


def main() -> None:
    from src.rag.rerank import get_reranker
    from src.rag.retrieve import get_retriever

    golden = load_golden(hard=True)
    r, rr = get_retriever(), get_reranker()

    print(f"computing once: dense search and reranker logits for "
          f"{len(golden)} questions...")
    rows = []
    for g in golden:
        hits = r.search(g["question"], k=50)
        gap = float(hits[0].score - hits[1].score)
        logits = rr.score(g["question"], [h.text for h in hits[:WINDOW]])
        order = list(np.argsort(-logits))
        rows.append({"gold": g["gold_chunk_id"],
                     "dense": [h.chunk_id for h in hits],
                     "reranked": [hits[i].chunk_id for i in order]
                                 + [h.chunk_id for h in hits[WINDOW:]],
                     "gap": gap})

    gaps = sorted(x["gap"] for x in rows)
    print(f"confidence gap: median {gaps[len(gaps)//2]:.3f}, "
          f"min {gaps[0]:.3f}, max {gaps[-1]:.3f}\n")

    out = []
    for th in THRESHOLDS:
        hit1 = hit5 = n_rr = 0
        for x in rows:
            use_dense = x["gap"] >= th          # confident -> skip the reranker
            ranked = x["dense"] if use_dense else x["reranked"]
            n_rr += not use_dense
            if x["gold"] in ranked[:1]:
                hit1 += 1
            if x["gold"] in ranked[:5]:
                hit5 += 1
        n = len(rows)
        frac = n_rr / n
        lat = DENSE_MS + frac * RERANK_MS
        out.append({"threshold": th, "recall1": hit1 / n, "recall5": hit5 / n,
                    "frac_reranked": frac, "p50_est_ms": lat})

    (EVALS / "rerank_gate.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                            encoding="utf-8")

    print("CONDITIONAL RERANKING: threshold on the confidence gap")
    hdr = (f"{'thresh':<8}{'reranked':>13}{'recall@1':>11}{'recall@5':>11}"
           f"{'est. p50':>12}")
    print(hdr); print("-" * len(hdr))
    for x in out:
        # The rule: rerank when the gap is SMALLER than the threshold. So a
        # threshold of 0 means "never", and a threshold above the largest gap
        # means "always". The first version labelled these rows the other way.
        tag = ""
        if x["frac_reranked"] == 0.0:
            tag = "  never reranks"
        elif x["frac_reranked"] == 1.0:
            tag = "  always reranks"
        print(f"{x['threshold']:<8.2f}{x['frac_reranked']*100:>12.0f}%"
              f"{x['recall1']:>11.3f}{x['recall5']:>11.3f}"
              f"{x['p50_est_ms']:>10.0f}ms{tag}")
    print("\nestimated p50 = 26 ms of dense search + share reranked x 2100 ms")


if __name__ == "__main__":
    main()
