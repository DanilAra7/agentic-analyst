"""Замер: сколько на самом деле дал бы иерархический поиск.

Зачем. В обсуждении прозвучало, что потолок иерархического поиска — lenient@20
= 0.980. Это НЕ так: lenient считается по чанкам (документ засчитан, если наверх
пробился любой его кусок), а иерархия ищет по ОДНОМУ вектору на документ.
Для тарифной таблицы в тысячи токенов такой вектор усредняет двадцать разных
разделов и может размазаться. Лучший кусок пробьётся, средний по документу - нет.

Здесь мы это проверяем прямо: считаем вектор каждого документа целиком и меряем,
как часто нужный ДОКУМЕНТ попадает в топ-k. Сравниваем с тем, что даёт нынешний
поиск по чанкам в мягкой (lenient) постановке - это честная пара.

Стоимость эксперимента - минуты. Стоимость постройки иерархии - день.
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

    print(f"считаем {len(texts)} векторов документов "
          f"(самый длинный {max(lens.values()):,} слов)...", flush=True)
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

    for hard, tag, title in ((False, "easy", "ЛЁГКИЙ"), (True, "hard", "ТРУДНЫЙ")):
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

        print(f"\n=== {title}  (вопросов: {n})")
        print(f"{'':26}" + "".join(f"@{k:<7}" for k in KS) + "   MRR")
        print(f"{'поиск по документам':26}" + "".join(f"{rec[k]/n:<8.3f}" for k in KS)
              + f"  {rr/n:.3f}")
        print(f"{'по чанкам, мягко (lenient)':26}"
              + "".join(f"{base['recall_lenient'][str(k)]:<8.3f}" for k in KS))
        print(f"{'по чанкам, строго (strict)':26}"
              + "".join(f"{base['recall_strict'][str(k)]:<8.3f}" for k in KS)
              + f"  {base['mrr']:.3f}")
        if misses:
            print(f"\nпровалы@5 ({len(misses)}), длина документа:")
            for d, w, r in sorted(misses, key=lambda x: -x[1])[:6]:
                print(f"  {d:<18} {w:>6,} слов   позиция: {r or 'вне топ-20'}")

    (EVALS / "ablation_doc_level.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nзаписано: {EVALS/'ablation_doc_level.json'}")


if __name__ == "__main__":
    main()
