"""Трассировка: где на самом деле уходит время и токены.

Зачем свой формат, а не сразу Langfuse. Числа в README должны воспроизводиться
у любого, кто склонировал репозиторий, без регистрации во внешнем сервисе.
Поэтому основа - JSONL на диск, а Langfuse подключается поверх как экспортёр.
Так у нас и воспроизводимость, и настоящий инструмент, а не выбор одного из двух.

Устройство простое: вложенные интервалы (spans) с именем, видом, длительностью
и произвольными полями. Текущая трасса лежит в contextvar, чтобы инструменты
могли отмечаться, не протаскивая объект через десять сигнатур.

Что важно мерить отдельно. Внутри одного вызова `search_docs` живут ДВА разных
по стоимости этапа: плотный поиск (миллисекунды) и cross-encoder (секунды).
Не разделив их, получим правдивое и бесполезное «поиск занял 2.5 с».
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
        """СОБСТВЕННОЕ время каждого интервала: своё минус время прямых детей.

        Без этого бюджет не сходится: `search_docs` длится 2.5 с, но 2.4 из них
        это вложенный реранкер, и в сумме по видам они посчитаются дважды.
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
        """Бюджет по видам: суммируется собственное время, поэтому части
        складываются в общее время трассы, а не превышают его."""
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
    if tr is None:                      # трассировка выключена - ничего не стоит
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
    """Дописать поля в последний открытый интервал."""
    tr = _CURRENT.get()
    if tr and tr.spans:
        tr.spans[-1].attrs.update(attrs)


def dump(traces: list[Trace], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False, default=str) + "\n")


def render(tr: Trace, max_rows: int = 40) -> str:
    """Дерево интервалов текстом: нужно для разбора одного провала глазами."""
    out = [f"trace {tr.trace_id}  {tr.scheme}/{tr.label}  всего {tr.total_ms:.0f} мс"]
    for sp, own in zip(tr.spans[:max_rows], tr.self_ms()[:max_rows]):
        pad = "  " * sp.depth
        keep = ("model", "cached", "rows", "chars", "window", "k", "tool_calls")
        extra = " ".join(f"{k}={sp.attrs[k]}" for k in keep if k in sp.attrs)
        out.append(f"  {sp.ms:>7.0f} мс (своё {own:>7.0f}) {pad}{sp.kind:<9} {sp.name:<14} {extra}")
    return "\n".join(out)
