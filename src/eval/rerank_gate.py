"""Условный реранк: не реранжировать, когда плотный поиск и так уверен.

Идея из бэклога №47(г), и она целится сразу в две проблемы, а не в одну.

  ЛАТЕНТНОСТЬ  реранк стоит 2.1 с при окне 20 против 26 мс без него.
               Если пропускать его там, где он не нужен, средняя цена падает.
  КАЧЕСТВО     решение №13: реранкер РОНЯЕТ recall@1 (0.765 -> 0.667), потому
               что предпочитает тематически цельный документ строке таблицы.
               Пропуская его на уверенных запросах, часть этой потери возвращаем.

Сигнал - РАЗРЫВ УВЕРЕННОСТИ: разница косинуса между первым и вторым кандидатом
плотного поиска. Большой разрыв означает, что поиск различает лидера; мелкий -
что кандидаты слиплись и порядок случаен, вот там реранкер и нужен.

Порог НЕ назначается на глаз: перебираем и смотрим, что происходит с обеими
метриками и с долей реранжированных запросов.

Считается локально. Логиты реранкера считаются ОДИН раз для всех вопросов,
дальше перебор порогов бесплатен.
"""
from __future__ import annotations

import json

import numpy as np

from src.config import EVALS
from src.eval.retrieval import load_golden

WINDOW = 20
DENSE_MS, RERANK_MS = 26.0, 2100.0     # замерено в src/eval/rerank_cost.py
THRESHOLDS = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 1.0)


def main() -> None:
    from src.rag.rerank import get_reranker
    from src.rag.retrieve import get_retriever

    golden = load_golden(hard=True)
    r, rr = get_retriever(), get_reranker()

    print(f"считаем один раз: плотный поиск и логиты реранкера для "
          f"{len(golden)} вопросов...")
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
    print(f"разрыв уверенности: медиана {gaps[len(gaps)//2]:.3f}, "
          f"мин {gaps[0]:.3f}, макс {gaps[-1]:.3f}\n")

    out = []
    for th in THRESHOLDS:
        hit1 = hit5 = n_rr = 0
        for x in rows:
            use_dense = x["gap"] >= th          # уверен -> не реранжируем
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

    print("УСЛОВНЫЙ РЕРАНК: порог по разрыву уверенности")
    hdr = (f"{'порог':<8}{'реранжируем':>13}{'recall@1':>11}{'recall@5':>11}"
           f"{'ожид. p50':>12}")
    print(hdr); print("-" * len(hdr))
    for x in out:
        # Правило: реранжируем, когда разрыв МЕНЬШЕ порога. Значит порог 0
        # означает «никогда», а порог выше максимального разрыва - «всегда».
        # Первая версия подписала эти строки наоборот.
        tag = ""
        if x["frac_reranked"] == 0.0:
            tag = "  реранк никогда"
        elif x["frac_reranked"] == 1.0:
            tag = "  реранк всегда"
        print(f"{x['threshold']:<8.2f}{x['frac_reranked']*100:>12.0f}%"
              f"{x['recall1']:>11.3f}{x['recall5']:>11.3f}"
              f"{x['p50_est_ms']:>10.0f}м{tag}")
    print("\nожидаемая p50 = 26 мс плотного поиска + доля реранжированных x 2100 мс")


if __name__ == "__main__":
    main()
