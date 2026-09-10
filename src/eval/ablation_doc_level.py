"""Measurement: what hierarchical retrieval would actually give.

Why. It was claimed in discussion that the ceiling of hierarchical retrieval is
lenient@20 = 0.980. That is NOT so: lenient is computed over chunks (a document
counts if any of its pieces broke through), while a hierarchy searches ONE vector
per document. For a rate table of thousands of tokens such a vector averages twenty
different sections and may smear out. The best piece breaks through; the document
average does not.

Here we test that directly: compute a vector for each whole document and measure how
often the right DOCUMENT lands in the top-k. Compared against what the current
chunk-level retrieval gives in its lenient form - that is the fair pair.

Cost of the experiment: minutes. Cost of building the hierarchy: a day.
"""
from __future__ import annotations

import json

import numpy as np

from src.config import DATA, DOCS, EVALS, settings
from src.eval.retrieval import KS, load_golden
from src.rag.chunking import parse_frontmatter

DOC_EMB = DATA / "doc_embeddings.npy"
DOC_IDS = DATA / "doc_ids.json"


def build_doc_index(model):
    ids, texts, lens = [], [], {}
    for path in sorted(DOCS.glob("*.md")):
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        doc_id = meta.get("document_id", path.stem)
        ids.append(doc_id)
        texts.append(f"{meta.get('title','')} [{doc_id}]\n\n{body}")
        lens[doc_id] = len(body.split())

    print(f"computing {len(texts)} document vectors "
          f"(longest is {max(lens.values()):,} words)...", flush=True)
    vecs = model.encode(texts, batch_size=4, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    np.save(DOC_EMB, vecs)
    DOC_IDS.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    return vecs, ids, lens


def main() -> None:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embed_model)
    vecs, ids, lens = build_doc_index(model)
    results = {}

    for hard, tag, title in ((False, "easy", "EASY"), (True, "hard", "HARD")):
        golden = load_golden(hard)
        q = model.encode([g["question"] for g in golden], normalize_embeddings=True,
                         convert_to_numpy=True, batch_size=32).astype(np.float32)
        order = np.argsort(-(vecs @ q.T), axis=0)

        rec = {k: 0 for k in KS}
        rr, misses = 0.0, []
        for j, g in enumerate(golden):
            ranked = [ids[i] for i in order[: max(KS), j]]
            r = ranked.index(g["gold_doc_id"]) + 1 if g["gold_doc_id"] in ranked else None
            for k in KS:
                rec[k] += r is not None and r <= k
            rr += 1.0 / r if r else 0.0
            if not (r and r <= 5):
                misses.append((g["gold_doc_id"], lens.get(g["gold_doc_id"], 0), r))

        n = len(golden)
        results[tag] = {"n": n, "recall": {k: rec[k] / n for k in KS}, "mrr": rr / n}
        base = json.loads((EVALS / "baseline_dense.json").read_text(encoding="utf-8"))[tag]

        print(f"\n=== {title}  (questions: {n})")
        print(f"{'':26}" + "".join(f"@{k:<7}" for k in KS) + "   MRR")
        print(f"{'document-level search':26}" + "".join(f"{rec[k]/n:<8.3f}" for k in KS)
              + f"  {rr/n:.3f}")
        print(f"{'chunk-level, lenient':26}"
              + "".join(f"{base['recall_lenient'][str(k)]:<8.3f}" for k in KS))
        print(f"{'chunk-level, strict':26}"
              + "".join(f"{base['recall_strict'][str(k)]:<8.3f}" for k in KS)
              + f"  {base['mrr']:.3f}")
        if misses:
            print(f"\nmisses@5 ({len(misses)}), document length:")
            for d, w, r in sorted(misses, key=lambda x: -x[1])[:6]:
                print(f"  {d:<18} {w:>6,} words   rank: {r or 'outside top-20'}")

    (EVALS / "ablation_doc_level.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten: {EVALS/'ablation_doc_level.json'}")


if __name__ == "__main__":
    main()
