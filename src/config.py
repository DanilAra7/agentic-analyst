"""Конфигурация из переменных окружения."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
DOCS = DATA / "docs"
EVALS = ROOT / "evals"
DB_PATH = DATA / "olist.duckdb"
CACHE_PATH = ROOT / "llm_cache.sqlite"


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq")
    llm_model: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    judge_provider: str = os.getenv("JUDGE_PROVIDER", "mistral")
    judge_model: str = os.getenv("JUDGE_MODEL", "mistral-small-latest")
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
    rerank_model: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

    def __post_init__(self) -> None:
        if (self.judge_provider, self.judge_model) == (self.llm_provider, self.llm_model):
            raise ValueError(
                "Судья обязан отличаться от генератора: модель систематически "
                "завышает оценку собственным ответам (self-preference bias)."
            )


settings = Settings()
