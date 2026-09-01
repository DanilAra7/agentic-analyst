"""Ablation: cross-encoder реранкер поверх плотного поиска.

Что меряем. Плотный поиск достаёт MAX_WINDOW кандидатов; реранкер
переупорядочивает первые W из них, хвост остаётся как был. Меняя W, получаем
кривую «качество против латентности».

Почему хвост не выбрасываем. Иначе recall@10 и recall@20 при W=10 стали бы
неопределимы, и строки таблицы перестали бы быть сравнимыми между собой.

Потолок. При окне W recall@k реранкера не может превысить recall@W плотного
поиска: чего нет в кандидатах, того не отранжируешь. Этот потолок печатается
рядом с результатом — чтобы было видно, упёрлись мы в ранжирование или в отбор.

Порядок этапов важен для памяти. Сначала ОБА набора кандидатов добываются
би-энкодером, затем би-энкодер выгружается, и только потом грузится реранкер.
Держать в памяти две модели по 2.3 ГБ одновременно незачем.

Оценка чисто оффлайновая: ни одного вызова LLM-API.
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
LAT_SAMPLE = 8          # сколько запросов таймим по-настоящему, по одному


def rss_gb() -> float:
    """Пиковое потребление памяти процессом. На macOS ru_maxrss в байтах."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3


def rerank_scores(reranker, texts_by_id: dict, golden: list[dict],
                  cand: list[list[str]], tag: str) -> list[list[float]]:
    """Логиты для всех пар (вопрос, кандидат). Кешируются на диск: прогон
    дорогой, а окна W вложены друг в друга и считаются из одних и тех же чисел."""
    path = EVALS / f"rerank_scores_{tag}.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached["questions"] == [g["question"] for g in golden] and cached["cand"] == cand:
            print(f"  [{tag}] логиты взяты из кеша {path.name}")
            return cached["scores"]

    t0 = time.perf_counter()
    out = []
    for i, (g, ids) in enumerate(zip(golden, cand), 1):
        out.append(reranker.score(g["question"], [texts_by_id[c] for c in ids]).tolist())
        if i % 10 == 0 or i == len(golden):
            el = time.perf_counter() - t0
            eta = el / i * (len(golden) - i)
            print(f"  [{tag}] {i}/{len(golden)}  прошло {el:.0f} c, осталось ~{eta:.0f} c, "
                  f"память {rss_gb():.1f} ГБ", flush=True)
    path.write_text(json.dumps(
        {"questions": [g["question"] for g in golden], "cand": cand, "scores": out},
        ensure_ascii=False), encoding="utf-8")
    return out


def apply_window(cand: list[str], scores: list[float], w: int) -> list[str]:
    """Переупорядочить первые w кандидатов по логиту, хвост оставить как есть."""
    w = min(w, len(cand))
    head = np.argsort(-np.asarray(scores[:w], dtype=np.float32))
    return [cand[i] for i in head] + cand[w:]


def measure_latency(reranker, texts_by_id: dict, golden: list[dict],
                    cand: list[list[str]], w: int) -> dict:
    """Честная латентность: по одному запросу, как в проде, а не батчем."""
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

    # --- этап 1: кандидаты. Тут нужен только би-энкодер ---
    retriever = get_retriever()
    sets = {}
    for hard, tag in ((False, "easy"), (True, "hard")):
        golden = load_golden(hard)
        sets[tag] = (golden, retriever.search_batch([g["question"] for g in golden],
                                                    k=MAX_WINDOW))
    texts_by_id = {k: v["text"] for k, v in retriever.by_id.items()}

    # --- выгружаем би-энкодер, он своё отработал ---
    del retriever.model
    get_retriever.cache_clear()
    del retriever
    free_memory()
    print(f"би-энкодер выгружен, пик памяти {rss_gb():.1f} ГБ\n")

    # --- этап 2: реранк ---
    reranker = get_reranker()
    print(f"реранкер: {reranker.model_name}\n  устройство {reranker.device}, "
          f"тип {reranker.dtype}, батч {reranker.batch_size}\n")

    results = {}
    for tag, title in (("easy", "ЛЁГКИЙ"), ("hard", "ТРУДНЫЙ")):
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

        print(f"\n=== {title}   (вопросов: {base['n']})")
        hdr = (f"{'вариант':<13}" + "".join(f"@{k:<7}" for k in KS)
               + f"{'MRR':>8}{'p50 мс':>9}{'потолок':>9}")
        print(hdr)
        print("-" * len(hdr))
        for name, r in rows.items():
            if name == "misses":
                continue
            lat = f"{r['latency']['p50_ms']:>9.0f}" if "latency" in r else f"{'—':>9}"
            ceil = f"{r['ceiling']:>9.3f}" if r.get("ceiling") is not None else f"{'—':>9}"
            print(f"{name:<13}"
                  + "".join(f"{r['recall_strict'][k]:<8.3f}" for k in KS)
                  + f"{r['mrr']:>8.3f}{lat}{ceil}", flush=True)

    (EVALS / "ablation_rerank.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nпик памяти за прогон: {rss_gb():.1f} ГБ")
    print(f"записано: {EVALS / 'ablation_rerank.json'}")


if __name__ == "__main__":
    main()
