"""Types and the LLM provider protocol."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

Message = dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)
    """Provider-specific fields that must be sent back unchanged.

    An OpenAI-compatible endpoint is compatible in FORM but not in semantics.
    Gemini 3 puts `thought_signature` here and requires it in the next request,
    answering 400 otherwise. Without passing this through, an agent loop is
    impossible in principle - and a single call never exposes the defect, which
    is why it only surfaced on the second round.
    """


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMProvider(Protocol):
    """One interface. Changing provider = changing one line in .env."""

    name: str
    model: str

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse: ...
