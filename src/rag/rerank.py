"""Реранкер: cross-encoder поверх кандидатов плотного поиска.

Зачем. Би-энкодер (bge-m3) кодирует вопрос и чанк НЕЗАВИСИМО, поэтому вектор
чанка можно посчитать заранее — отсюда скорость. Ценой этого чанк не знает,
о чём его спросят: один вектор обязан обслуживать все возможные вопросы.

Cross-encoder читает пару (вопрос, чанк) СОВМЕСТНО, одним проходом трансформера,
и токены вопроса видят токены чанка через attention. Это точнее, но считать
заранее нечего: стоимость = K прогонов модели на каждый запрос.

Отсюда единственная разумная схема — каскад: дешёвый поиск отбирает K кандидатов,
дорогой реранкер их переупорядочивает. K — главная ручка размена
качество/латентность, и именно её мы меряем в ablation.

О ПАМЯТИ (измерено, а не предположено). Модель — XLM-RoBERTa-large, 568M
параметров, 2.3 ГБ в float32. Первый прогон держал в памяти ОДНОВРЕМЕННО
би-энкодер и реранкер в float32 и дорос до 10 ГБ на машине с 16 ГБ: система
ушла в своп, GPU стал ждать диск, счёт остановился совсем.
Что помогло: float16 + выгрузка би-энкодера после отбора кандидатов.
Пик упал 10 ГБ -> 3.6 ГБ.
Что НЕ помогло: размер батча. Замер 8/16/32/50 на окне 50 дал 2119/2101/2110/2126 мс
— разброс в пределах шума. GPU упирается в вычисление, а не в накладные расходы
на запуск, поэтому батч тут не ручка. Оставлен 16 как компромисс.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from src.config import settings

BATCH_SIZE = 16      # см. блок «О ПАМЯТИ»: на скорость не влияет
MAX_LENGTH = 512     # чанки до 400 токенов + вопрос — влезает без обрезки


def free_memory() -> None:
    """Освободить кеш ускорителя. Нужно между этапами каскада: держать
    би-энкодер в памяти во время реранка незачем, он своё уже отработал."""
    import torch

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class Reranked:
    chunk_id: str
    score: float          # логит cross-encoder, НЕ сравним с косинусом би-энкодера
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
        """Сырые логиты релевантности. Монотонного преобразования (sigmoid) не
        делаем: для ранжирования важен порядок, а он от него не меняется."""
        if not texts:
            return np.empty(0, dtype=np.float32)
        pairs = [(query, t) for t in texts]
        s = self.model.predict(pairs, batch_size=self.batch_size,
                               show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(s, dtype=np.float32).ravel()

    def rerank(self, query: str, hits: list, top_k: int | None = None) -> list[Reranked]:
        """hits — объекты с полями chunk_id/text/meta (Hit из retrieve.py)."""
        scores = self.score(query, [h.text for h in hits])
        order = np.argsort(-scores)
        out = [Reranked(hits[i].chunk_id, float(scores[i]), hits[i].text, hits[i].meta)
               for i in order]
        return out[:top_k] if top_k else out


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()
