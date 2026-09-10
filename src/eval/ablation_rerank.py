"""Ablation: a cross-encoder reranker on top of dense search.

What is measured. Dense search pulls MAX_WINDOW candidates; the reranker reorders
the first W of them and the tail stays as it was. Varying W gives a
"quality against latency" curve.

Why the tail is not discarded. Otherwise recall@10 and recall@20 at W=10 would be
undefined, and the table rows would stop being comparable with each other.

The ceiling. At window W the reranker's recall@k cannot exceed dense search's
recall@W: you cannot rank what is not among the candidates. That ceiling is
printed next to the result - so it is visible whether we hit a limit of ranking
or a limit of selection.

The order of the stages matters for memory. First BOTH candidate sets are fetched
by the bi-encoder, then the bi-encoder is unloaded, and only then is the reranker
loaded. There is no reason to hold two 2.3 GB models in memory at once.

The evaluation is purely offline: not a single LLM API call.
"""
from __future__ import annotations

import json
import resource
import time

import numpy as np

from src.config import EVALS
from src.eval.retrieval import KS, load_golden, score

WINDOWS = (10, 20, 50)
MAX_WINDOW = max(WINDOWS)
LAT_SAMPLE = 8          # how many queries are really timed, one at a time


def rss_gb() -> float:
    """Peak memory used by the process. On macOS ru_maxrss is in bytes."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3


def rerank_scores(reranker, texts_by_id: dict, golden: list[dict],
                  cand: list[list[str]], tag: str) -> list[list[float]]:
    """Logits for every (question, candidate) pair. Cached to disk: the run is
    expensive, and the W windows are nested and computed from the same numbers."""
    path = EVALS / f"rerank_scores_{tag}.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached["questions"] == [g["question"] for g in golden] and cached["cand"] == cand:
            print(f"  [{tag}] logits taken from cache {path.name}")
            return cached["scores"]

    t0 = time.perf_counter()
    out = []
    for i, (g, ids) in enumerate(zip(golden, cand), 1):
        out.append(reranker.score(g["question"], [texts_by_id[c] for c in ids]).tolist())
        if i % 10 == 0 or i == len(golden):
            el = time.perf_counter() - t0
            eta = el / i * (len(golden) - i)
            print(f"  [{tag}] {i}/{len(golden)}  elapsed {el:.0f} s, ~{eta:.0f} s left, "
                  f"memory {rss_gb():.1f} GB", flush=True)
    path.write_text(json.dumps(
        {"questions": [g["question"] for g in golden], "cand": cand, "scores": out},
        ensure_ascii=False), encoding="utf-8")
    return out


def apply_window(cand: list[str], scores: list[float], w: int) -> list[str]:
    """Reorder the first w candidates by logit, leave the tail as it is."""
    w = min(w, len(cand))
    head = np.argsort(-np.asarray(scores[:w], dtype=np.float32))
    return [cand[i] for i in head] + cand[w:]


def measure_latency(reranker, texts_by_id: dict, golden: list[dict],
                    cand: list[list[str]], w: int) -> dict:
    """Honest latency: one query at a time, as in production, not batched."""
    lat = []
    for g, ids in list(zip(golden, cand))[:LAT_SAMPLE]:
        texts = [texts_by_id[c] for c in ids[:w]]
        t0 = time.perf_counter()
        reranker.score(g["question"], texts)
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    return {"p50_ms": lat[len(lat) // 2], "max_ms": lat[-1], "mean_ms": sum(lat) / len(lat)}


def main() -> None:
    from src.rag.rerank import free_memory, get_reranker
    from src.rag.retrieve import get_retriever

    # --- stage 1: candidates. Only the bi-encoder is needed here ---
    retriever = get_retriever()
    sets = {}
    for hard, tag in ((False, "easy"), (True, "hard")):
        golden = load_golden(hard)
        sets[tag] = (golden, retriever.search_batch([g["question"] for g in golden],
                                                    k=MAX_WINDOW))
    texts_by_id = {k: v["text"] for k, v in retriever.by_id.items()}

    # --- unload the bi-encoder, its work is done ---
    del retriever.model
    get_retriever.cache_clear()
    del retriever
    free_memory()
    print(f"bi-encoder unloaded, peak memory {rss_gb():.1f} GB\n")

    # --- stage 2: reranking ---
    reranker = get_reranker()
    print(f"reranker: {reranker.model_name}\n  device {reranker.device}, "
          f"dtype {reranker.dtype}, batch {reranker.batch_size}\n")

    results = {}
    for tag, title in (("easy", "EASY"), ("hard", "HARD")):
        golden, cand = sets[tag]
        base = score(cand, golden)
        logits = rerank_scores(reranker, texts_by_id, golden, cand, tag)

        rows = {"dense": {k: v for k, v in base.items() if k != "misses"}}
        for w in WINDOWS:
            ranked = [apply_window(c, s, w) for c, s in zip(cand, logits)]
            r = score(ranked, golden)
            r["latency"] = measure_latency(reranker, texts_by_id, golden, cand, w)
            r["ceiling"] = base["recall_strict"].get(w)
            rows[f"rerank_w{w}"] = {k: v for k, v in r.items() if k != "misses"}
            if w == WINDOWS[-1]:
                rows["misses"] = r["misses"]
        results[tag] = rows

        print(f"\n=== {title}   (questions: {base['n']})")
        hdr = (f"{'variant':<13}" + "".join(f"@{k:<7}" for k in KS)
               + f"{'MRR':>8}{'p50 ms':>9}{'ceiling':>9}")
        print(hdr)
        print("-" * len(hdr))
        for name, r in rows.items():
            if name == "misses":
                continue
            lat = f"{r['latency']['p50_ms']:>9.0f}" if "latency" in r else f"{'-':>9}"
            ceil = f"{r['ceiling']:>9.3f}" if r.get("ceiling") is not None else f"{'-':>9}"
            print(f"{name:<13}"
                  + "".join(f"{r['recall_strict'][k]:<8.3f}" for k in KS)
                  + f"{r['mrr']:>8.3f}{lat}{ceil}", flush=True)

    (EVALS / "ablation_rerank.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\npeak memory for the run: {rss_gb():.1f} GB")
    print(f"written: {EVALS / 'ablation_rerank.json'}")


if __name__ == "__main__":
    main()
