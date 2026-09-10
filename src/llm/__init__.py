from __future__ import annotations

from src.config import settings
from src.llm.base import LLMResponse, Message, ToolCall
from src.llm.providers import OpenAICompatProvider


def get_llm(provider: str | None = None, model: str | None = None) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        provider or settings.llm_provider, model or settings.llm_model
    )


def get_judge() -> OpenAICompatProvider:
    """The judge must be a different model from the generator (see Settings.__post_init__)."""
    return OpenAICompatProvider(settings.judge_provider, settings.judge_model)


__all__ = ["get_llm", "get_judge", "LLMResponse", "Message", "ToolCall"]
