"""Реранкер: качество против латентности по размеру окна.

Зачем отдельный замер. Из M2 (решение №11) известно КАЧЕСТВО при окнах 10/20/50.
Из M4 (решение №32) известно, что реранкер съедает 98% времени поиска. Не хватало
пары: сколько стоит каждое окно. Без неё выбор K делается на глаз.

Считается локально, без единого вызова провайдера, и потому воспроизводится
всегда. Прогрев обязателен: первый вызов тянет с диска модель на 568M параметров,
и без прогрева в замер попадёт загрузка весов, а не работа.
"""
from __future__ import annotations

import json
import time

from src.config import EVALS
from src.eval.retrieval import load_golden

WINDOWS = (0, 5, 10, 20, 50)      # 0 = реранкер выключен
REPEATS = 3

# recall@5 и recall@1 на трудном наборе, замерено в M2 (evals/ablation_rerank.json).
# Здесь не пересчитывается: цель этого файла - ЦЕНА, качество уже известно.
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
    rr.score("warmup", ["a"])                       # прогрев обеих моделей

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
        print(f"  окно {w:>2}: p50 {pct(lat,50):>7.0f} мс   p95 {pct(lat,95):>7.0f} мс",
              flush=True)

    (EVALS / "rerank_cost.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                            encoding="utf-8")

    print("=" * 76)
    print("КАЧЕСТВО ПРОТИВ ЦЕНЫ  (трудный набор, 51 вопрос)")
    hdr = (f"{'окно':<7}{'recall@1':>10}{'recall@5':>10}{'p50 поиска':>13}"
           f"{'p95':>10}{'цена @5':>17}")
    print(hdr); print("-" * len(hdr))
    base = rows[0]
    for x in rows:
        if "r5" not in x:
            print(f"{x['window']:<7}{'—':>10}{'—':>10}{x['p50']:>12.0f}м{x['p95']:>9.0f}м")
            continue
        d5 = x["r5"] - base["r5"]
        dms = x["p50"] - base["p50"]
        cost = f"{dms/(d5*100):.0f} мс/пункт" if d5 > 0 else "—"
        print(f"{x['window']:<7}{x['r1']:>10.3f}{x['r5']:>10.3f}"
              f"{x['p50']:>12.0f}м{x['p95']:>9.0f}м{cost:>17}")
    print()
    print("«цена @5» - сколько миллисекунд стоит один процентный пункт recall@5")
    print("относительно выключенного реранкера.")


if __name__ == "__main__":
    main()
