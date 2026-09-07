"""Экспорт собственных трасс в Langfuse.

Почему экспортёр, а не прямая инструментация Langfuse. Числа в README должны
воспроизводиться у любого, кто склонировал репозиторий, БЕЗ регистрации во
внешнем сервисе (решение №31). Поэтому источник истины - `evals/traces.jsonl`,
а Langfuse получает копию. Побочная выгода: экспортировать можно задним числом,
в том числе трассы, снятые до того, как появился аккаунт.

Соответствие понятий:
    наша Trace          -> корневое наблюдение типа agent
    span kind=llm       -> generation (Langfuse считает по ней токены и стоимость)
    span kind=tool      -> tool
    span kind=retrieval -> retriever
    span kind=rerank    -> span
Вложенность восстанавливается из поля `depth`: интервалы лежат плоским списком
в порядке открытия, так что стек глубин однозначно задаёт дерево.

Времена берутся из трассы, а не из момента экспорта: иначе в Langfuse попадёт
длительность самого экспорта, а не измеренная.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from src.config import EVALS

KIND_TO_TYPE = {"llm": "generation", "tool": "tool",
                "retrieval": "retriever", "rerank": "span"}


def _host() -> str:
    return (os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL")
            or "https://cloud.langfuse.com")


def _client():
    from langfuse import Langfuse

    pk, sk = os.getenv("LANGFUSE_PUBLIC_KEY"), os.getenv("LANGFUSE_SECRET_KEY")
    if not (pk and sk):
        raise SystemExit(
            "Нет LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY в .env.\n"
            "Ключи заводятся в проекте на cloud.langfuse.com (Settings -> API keys).\n"
            "Посмотреть, что ушло бы, не отправляя: DRY_RUN=1 make langfuse")
    # Langfuse разводит регионы разными адресами (eu / us), и SDK читает
    # LANGFUSE_HOST. В .env переменную часто называют LANGFUSE_BASE_URL -
    # принимаем оба имени, иначе трассы молча уедут не в тот регион.
    c = Langfuse(public_key=pk, secret_key=sk, host=_host())
    if not c.auth_check():
        raise SystemExit("Langfuse отверг ключи: проверь пару public/secret и host.")
    return c


def load(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Нет {path}. Сначала собери трассы: LLM_CACHE=0 make latency")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _tree(spans: list[dict]) -> list[tuple[dict, int]]:
    """(интервал, индекс родителя). Родитель - ближайший предыдущий интервал
    с меньшей глубиной; -1 означает корень."""
    out, stack = [], []
    for i, sp in enumerate(spans):
        while stack and spans[stack[-1]]["depth"] >= sp["depth"]:
            stack.pop()
        out.append((sp, stack[-1] if stack else -1))
        stack.append(i)
    return out


def export_one(client, tr: dict) -> None:
    """Отправить одну трассу.

    ОГРАНИЧЕНИЕ, которое надо назвать вслух. В Langfuse v4 наблюдение начинается
    «сейчас»: `start_observation` не принимает время старта, задать можно только
    `end_time`. Значит абсолютные метки времени в интерфейсе будут временем
    ЭКСПОРТА, а не измерения. ДЛИТЕЛЬНОСТИ при этом сохраняются точно - именно
    они и нужны для разбора. Настоящий момент замера кладём в метаданные, чтобы
    он не потерялся.

    Порядок закрытия важен: дети закрываются РАНЬШЕ родителей, иначе вложенность
    в интерфейсе развалится. Поэтому создаём в прямом порядке, закрываем в обратном.
    """
    root = client.start_observation(
        name=f"{tr['scheme']}:{tr['label']}", as_type="agent",
        input={"label": tr["label"]},
        metadata={"scheme": tr["scheme"], "total_ms": round(tr["total_ms"]),
                  "measured_at": datetime.fromtimestamp(tr["started_at"],
                                                        tz=timezone.utc).isoformat(),
                  "trace_id_local": tr["trace_id"],
                  **{k: v for k, v in (tr.get("outcome") or {}).items()}})

    made: list = []
    for sp, parent in _tree(tr["spans"]):
        attrs = sp.get("attrs") or {}
        kind = KIND_TO_TYPE.get(sp["kind"], "span")
        owner = made[parent][0] if parent >= 0 else root
        kw = {"name": sp["name"], "as_type": kind, "metadata": attrs}
        if kind == "generation":
            kw["model"] = attrs.get("model")
            kw["usage_details"] = {"input": attrs.get("prompt_tokens", 0),
                                   "output": attrs.get("completion_tokens", 0)}
        made.append((owner.start_observation(**kw), sp["ms"]))

    for o, ms in reversed(made):
        o.end(end_time=time.time_ns() + int(ms * 1e6))
    root.end(end_time=time.time_ns() + int(tr["total_ms"] * 1e6))


def dry_run(traces: list[dict]) -> None:
    """Показать, что ушло бы, не отправляя ничего. Нужно, чтобы проверить
    сборку дерева и разметку типов без аккаунта."""
    print("DRY RUN: ничего не отправляется\n")
    for tr in traces[:3]:
        print(f"agent  {tr['scheme']}:{tr['label']}   {tr['total_ms']:.0f} мс")
        for sp, parent in _tree(tr["spans"]):
            pad = "  " * (sp["depth"] + 1)
            t = KIND_TO_TYPE.get(sp["kind"], "span")
            print(f"{pad}{t:<11}{sp['name']:<14}{sp['ms']:>8.0f} мс"
                  f"   родитель: {'корень' if parent < 0 else tr['spans'][parent]['name']}")
        print()
    kinds: dict[str, int] = {}
    for tr in traces:
        for sp in tr["spans"]:
            k = KIND_TO_TYPE.get(sp["kind"], "span")
            kinds[k] = kinds.get(k, 0) + 1
    print(f"всего трасс: {len(traces)}   наблюдений по типам: {kinds}")


def main() -> None:
    path = EVALS / "traces.jsonl"
    traces = load(path)
    if os.getenv("DRY_RUN") == "1":
        dry_run(traces)
        return

    client = _client()
    for tr in traces:
        export_one(client, tr)
    client.flush()
    print(f"отправлено трасс: {len(traces)}")
    print(f"смотреть: {_host()}")


if __name__ == "__main__":
    main()
