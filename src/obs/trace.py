"""Tracing: where the time and the tokens actually go.

Why an own format rather than Langfuse directly. The numbers in the README must be
reproducible by anyone who clones the repository, without registering with an
external service. So the base is JSONL on disk, and Langfuse plugs in on top as an
exporter. That gives both reproducibility and a real tool, instead of a choice
between the two.

The design is simple: nested spans with a name, a kind, a duration and arbitrary
fields. The current trace lives in a contextvar so that tools can record themselves
without dragging the object through ten signatures.

What matters to measure separately. Inside a single `search_docs` call live TWO
stages of wildly different cost: dense retrieval (milliseconds) and the
cross-encoder (seconds). Without separating them you get a true and useless
"search took 2.5 s".
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_CURRENT: ContextVar["Trace | None"] = ContextVar("current_trace", default=None)


@dataclass
class Span:
    name: str
    kind: str                      # llm | tool | retrieval | rerank
    ms: float = 0.0
    depth: int = 0
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    label: str = ""
    scheme: str = ""
    started_at: float = field(default_factory=time.time)
    total_ms: float = 0.0
    spans: list[Span] = field(default_factory=list)
    outcome: dict = field(default_factory=dict)
    _depth: int = 0

    def self_ms(self) -> list[float]:
        """SELF time of every span: its own minus the time of its direct children.

        Without this the budget does not add up: `search_docs` lasts 2.5 s, but 2.4
        of them are the nested reranker, and summing by kind counts them twice.
        """
        out = []
        for i, sp in enumerate(self.spans):
            child = 0.0
            for nxt in self.spans[i + 1:]:
                if nxt.depth <= sp.depth:
                    break
                if nxt.depth == sp.depth + 1:
                    child += nxt.ms
            out.append(max(sp.ms - child, 0.0))
        return out

    def by_kind(self) -> dict[str, float]:
        """Budget by kind: self time is summed, so the parts add up to the trace's
        total rather than exceeding it."""
        out: dict[str, float] = {}
        for sp, own in zip(self.spans, self.self_ms()):
            out[sp.kind] = out.get(sp.kind, 0.0) + own
        return out

    def tokens(self) -> tuple[int, int]:
        p = sum(s.attrs.get("prompt_tokens", 0) for s in self.spans)
        c = sum(s.attrs.get("completion_tokens", 0) for s in self.spans)
        return p, c

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_depth", None)
        return d


@contextmanager
def trace(label: str = "", scheme: str = ""):
    tr = Trace(label=label, scheme=scheme)
    token = _CURRENT.set(tr)
    t0 = time.perf_counter()
    try:
        yield tr
    finally:
        tr.total_ms = (time.perf_counter() - t0) * 1000
        _CURRENT.reset(token)


@contextmanager
def span(name: str, kind: str, **attrs):
    tr = _CURRENT.get()
    if tr is None:                      # tracing is off - costs nothing
        yield None
        return
    s = Span(name=name, kind=kind, depth=tr._depth, attrs=attrs)
    tr.spans.append(s)
    tr._depth += 1
    t0 = time.perf_counter()
    try:
        yield s
    finally:
        s.ms = (time.perf_counter() - t0) * 1000
        tr._depth -= 1


def current() -> Trace | None:
    return _CURRENT.get()


def note(**attrs) -> None:
    """Add fields to the most recently opened span."""
    tr = _CURRENT.get()
    if tr and tr.spans:
        tr.spans[-1].attrs.update(attrs)


def dump(traces: list[Trace], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False, default=str) + "\n")


def render(tr: Trace, max_rows: int = 40) -> str:
    """The span tree as text: for examining a single failure by eye."""
    out = [f"trace {tr.trace_id}  {tr.scheme}/{tr.label}  total {tr.total_ms:.0f} ms"]
    for sp, own in zip(tr.spans[:max_rows], tr.self_ms()[:max_rows]):
        pad = "  " * sp.depth
        keep = ("model", "cached", "rows", "chars", "window", "k", "tool_calls")
        extra = " ".join(f"{k}={sp.attrs[k]}" for k in keep if k in sp.attrs)
        out.append(f"  {sp.ms:>7.0f} ms (self {own:>7.0f}) {pad}{sp.kind:<9} {sp.name:<14} {extra}")
    return "\n".join(out)
