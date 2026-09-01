"""Плотный поиск по эмбеддингам.

Точный перебор: эмбеддинги нормированы, поэтому косинус = скалярное
произведение, а поиск по всей базе = одно матричное умножение.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from src.config import settings
from src.rag.index import EMB_PATH, IDS_PATH, load_chunks


@dataclass
class Hit:
    chunk_id: str
    score: float
    text: str
    meta: dict


class DenseRetriever:
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.vectors: np.ndarray = np.load(EMB_PATH)
        self.ids: list[str] = json.loads(IDS_PATH.read_text(encoding="utf-8"))
        self.by_id = {c["chunk_id"]: c for c in load_chunks()}
        self.model = SentenceTransformer(settings.embed_model)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        q = self.model.encode([query], normalize_embeddings=True,
                              convert_to_numpy=True).astype(np.float32)
        scores = (self.vectors @ q.T).ravel()
        # argpartition вместо полной сортировки: нам нужен топ-k, а не порядок всех
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        out = []
        for i in top:
            cid = self.ids[i]
            c = self.by_id[cid]
            out.append(Hit(cid, float(scores[i]), c["text"], c["meta"]))
        return out

    def search_batch(self, queries: list[str], k: int = 20) -> list[list[str]]:
        """Батчем, для eval-прогонов: возвращает только id."""
        q = self.model.encode(queries, normalize_embeddings=True,
                              convert_to_numpy=True, batch_size=32).astype(np.float32)
        scores = self.vectors @ q.T          # (n_chunks, n_queries)
        k = min(k, scores.shape[0])
        idx = np.argpartition(-scores, k - 1, axis=0)[:k]
        out = []
        for j in range(scores.shape[1]):
            col = idx[:, j]
            col = col[np.argsort(-scores[col, j])]
            out.append([self.ids[i] for i in col])
        return out


@lru_cache(maxsize=1)
def get_retriever() -> DenseRetriever:
    return DenseRetriever()
