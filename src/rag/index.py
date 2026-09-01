"""Построение индекса: эмбеддинги чанков.

Поиск - точный, полным перебором по косинусу. Это осознанный выбор бейзлайна:
на 422 чанках точный перебор занимает миллисекунды, а приближённый поиск (HNSW)
мы добавим позже отдельной строкой ablation и измерим, сколько recall стоит
его скорость. Точный поиск при этом остаётся эталоном, с которым сравниваем.
"""
from __future__ import annotations

import json

import numpy as np

from src.config import DATA, EVALS, settings
from src.rag.chunking import CHUNKS_PATH

EMB_PATH = DATA / "chunk_embeddings.npy"
IDS_PATH = DATA / "chunk_ids.json"


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Нет {CHUNKS_PATH}. Запусти: uv run python -m src.rag.chunking")
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build() -> None:
    from sentence_transformers import SentenceTransformer

    chunks = load_chunks()
    model = SentenceTransformer(settings.embed_model)

    vectors = model.encode(
        [c["text"] for c in chunks],
        batch_size=16,
        normalize_embeddings=True,   # нормируем -> косинус = скалярное произведение
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    EVALS.mkdir(parents=True, exist_ok=True)
    np.save(EMB_PATH, vectors)
    IDS_PATH.write_text(
        json.dumps([c["chunk_id"] for c in chunks], ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n  чанков      {len(chunks):>6,}")
    print(f"  размерность {vectors.shape[1]:>6,}")
    print(f"  память      {vectors.nbytes / 1024**2:>6.1f} МБ")
    print(f"\n  {EMB_PATH}")


if __name__ == "__main__":
    build()
