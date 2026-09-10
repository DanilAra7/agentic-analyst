"""Reranker: a cross-encoder on top of dense retrieval candidates.

Why. A bi-encoder (bge-m3) encodes the question and the chunk INDEPENDENTLY, so
the chunk vector can be computed in advance - hence the speed. The price is that
the chunk does not know what it will be asked: one vector has to serve every
possible question.

A cross-encoder reads the pair (question, chunk) JOINTLY, in a single transformer
pass, and the question's tokens see the chunk's tokens through attention. That is
more accurate, but there is nothing to precompute: the cost is K model passes per
query.

Hence the only sensible scheme is a cascade: cheap retrieval selects K candidates,
the expensive reranker reorders them. K is the main quality/latency trade-off knob,
and it is exactly what the ablation measures.

ON MEMORY (measured, not assumed). The model is XLM-RoBERTa-large, 568M parameters,
2.3 GB in float32. The first run held the bi-encoder and the reranker in memory
SIMULTANEOUSLY in float32 and grew to 10 GB on a 16 GB machine: the system started
swapping, the GPU waited on disk, and the computation stopped entirely.
What helped: float16 plus unloading the bi-encoder after candidate selection.
The peak fell from 10 GB to 3.6 GB.
What did NOT help: batch size. Measuring 8/16/32/50 at window 50 gave
2119/2101/2110/2126 ms - a spread within noise. The GPU is compute-bound, not
launch-overhead-bound, so batch size is not a knob here. 16 was kept as a compromise.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from src.config import settings

BATCH_SIZE = 16      # see the ON MEMORY block: it does not affect speed
MAX_LENGTH = 512     # chunks up to 400 tokens plus the question: fits without truncation


def free_memory() -> None:
    """Free the accelerator cache. Needed between cascade stages: there is no
    point holding the bi-encoder in memory during reranking, its work is done."""
    import torch

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class Reranked:
    chunk_id: str
    score: float          # cross-encoder logit, NOT comparable to a bi-encoder cosine
    text: str
    meta: dict


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None, max_length: int = MAX_LENGTH,
                 batch_size: int = BATCH_SIZE, device: str | None = None,
                 fp16: bool = True) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        self.model_name = model_name or settings.rerank_model
        self.batch_size = batch_size
        kwargs: dict = {}
        if fp16 and (torch.backends.mps.is_available() or torch.cuda.is_available()):
            kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
        self.model = CrossEncoder(self.model_name, max_length=max_length,
                                  device=device, **kwargs)
        self.dtype = str(next(self.model.model.parameters()).dtype)
        self.device = str(self.model.model.device)

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        """Raw relevance logits. No monotone transform (sigmoid) is applied:
        ranking depends on order, and a monotone transform does not change it."""
        if not texts:
            return np.empty(0, dtype=np.float32)
        pairs = [(query, t) for t in texts]
        s = self.model.predict(pairs, batch_size=self.batch_size,
                               show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(s, dtype=np.float32).ravel()

    def rerank(self, query: str, hits: list, top_k: int | None = None) -> list[Reranked]:
        """hits - objects with chunk_id/text/meta fields (Hit from retrieve.py)."""
        scores = self.score(query, [h.text for h in hits])
        order = np.argsort(-scores)
        out = [Reranked(hits[i].chunk_id, float(scores[i]), hits[i].text, hits[i].meta)
               for i in order]
        return out[:top_k] if top_k else out


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()
