"""Retrieval quality metrics.

Computed from embeddings and gold labels - WITHOUT a single LLM call.
That makes the whole retrieval ablation free and runnable as often as needed.

Two levels of strictness:
  strict  - did exactly the chunk the question was generated from appear
  lenient - did any chunk of the same document appear

The gap between them shows whether we are losing on chunking: if lenient is much
higher than strict, the document is found but the wrong piece of it is picked.
"""
from __future__ import annotations

import json
from collections import defaultdict

from src.config import EVALS
from src.eval.golden import GOLDEN_HARD_PATH, GOLDEN_PATH

KS = (1, 3, 5, 10, 20)


def load_golden(hard: bool = False) -> list[dict]:
    path = GOLDEN_HARD_PATH if hard else GOLDEN_PATH
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def score(ranked_ids: list[list[str]], golden: list[dict]) -> dict:
    """ranked_ids[i] - the ordered list of chunk_ids for golden[i]."""
    strict = {k: 0 for k in KS}
    lenient = {k: 0 for k in KS}
    rr_strict = 0.0
    by_family = defaultdict(lambda: {"n": 0, "hit5": 0})
    misses = []

    for ranked, g in zip(ranked_ids, golden):
        docs = [cid.split("#")[0] for cid in ranked]
        s_rank = ranked.index(g["gold_chunk_id"]) + 1 if g["gold_chunk_id"] in ranked else None
        l_rank = docs.index(g["gold_doc_id"]) + 1 if g["gold_doc_id"] in docs else None

        for k in KS:
            strict[k] += s_rank is not None and s_rank <= k
            lenient[k] += l_rank is not None and l_rank <= k
        rr_strict += 1.0 / s_rank if s_rank else 0.0

        fam = by_family[g["family"]]
        fam["n"] += 1
        fam["hit5"] += s_rank is not None and s_rank <= 5
        if not (s_rank and s_rank <= 5):
            misses.append({**g, "rank": s_rank})

    n = len(golden)
    return {
        "n": n,
        "recall_strict": {k: strict[k] / n for k in KS},
        "recall_lenient": {k: lenient[k] / n for k in KS},
        "mrr": rr_strict / n,
        "by_family": {f: v["hit5"] / v["n"] for f, v in sorted(by_family.items())},
        "family_n": {f: v["n"] for f, v in sorted(by_family.items())},
        "misses": misses,
    }


def report(res: dict, title: str) -> None:
    print(f"\n=== {title}   (questions: {res['n']})")
    print(f"{'':10}" + "".join(f"@{k:<7}" for k in KS))
    print(f"{'strict':10}" + "".join(f"{res['recall_strict'][k]:<8.3f}" for k in KS))
    print(f"{'lenient':10}" + "".join(f"{res['recall_lenient'][k]:<8.3f}" for k in KS))
    print(f"\nMRR (strict): {res['mrr']:.3f}")
    print("\nrecall@5 by family:")
    for f, v in res["by_family"].items():
        print(f"  {f:<6} n={res['family_n'][f]:<4} {v:.3f}")


def main() -> None:
    from src.rag.retrieve import get_retriever

    r = get_retriever()
    results = {}
    for hard, title in ((False, "EASY (document wording)"),
                        (True,  "HARD (user wording)")):
        golden = load_golden(hard)
        ranked = r.search_batch([g["question"] for g in golden], k=max(KS))
        results[hard] = score(ranked, golden)
        report(results[hard], title)

    e, h = results[False], results[True]
    print("\n" + "=" * 62)
    print("COMPARISON: the price of the vocabulary gap")
    print(f"{'':12}{'easy':>10}{'hard':>10}{'delta':>10}")
    for k in KS:
        a, b = e["recall_strict"][k], h["recall_strict"][k]
        print(f"  recall@{k:<4}{a:>10.3f}{b:>10.3f}{b - a:>+10.3f}")
    print(f"  {'MRR':<10}{e['mrr']:>10.3f}{h['mrr']:>10.3f}{h['mrr'] - e['mrr']:>+10.3f}")
    print(f"\n{'':12}{'lenient@5':>10}")
    print(f"  {'easy':<10}{e['recall_lenient'][5]:>10.3f}")
    print(f"  {'hard':<10}{h['recall_lenient'][5]:>10.3f}")

    (EVALS / "baseline_dense.json").write_text(json.dumps(
        {"easy": {k: v for k, v in e.items() if k != "misses"},
         "hard": {k: v for k, v in h.items() if k != "misses"}},
        ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nmisses on the hard set: {len(h['misses'])} of {h['n']}")
    for m in h["misses"][:10]:
        pos = f"#{m['rank']}" if m["rank"] else "none"
        print(f"  [{m['family']:<3} {pos:>4}] {m['question'][:86]}")


if __name__ == "__main__":
    main()
