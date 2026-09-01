"""Чистка golden set от вопросов, на которые эталонный чанк не отвечает.

Дефект, найденный при проверке гипотезы: промпт требовал называть штат или
категорию ради однозначности, и модель называла их даже там, где чанк к
штату/категории отношения не имеет. Получился вопрос «когда придёт заказ
канцтоваров в Сан-Паулу», эталоном которому назначен категорийный справочник,
где Сан-Паулу не упоминается вовсе.

Такой вопрос неотвечаем как поставлен, и засчитывать его как провал поиска -
врать себе.
"""
from __future__ import annotations

import json

from src.eval.golden import GOLDEN_HARD_PATH, GOLDEN_PATH
from src.ingest.build_docs import CATEGORIES, STATES
from src.rag.index import load_chunks

STATE_NAMES = {n: c for c, n, _ in STATES}


def entity_supported(question: str, chunk: dict) -> tuple[bool, str]:
    """Если вопрос называет штат или категорию, чанк обязан их поддерживать."""
    q = question.lower()
    text = chunk["text"].lower()
    meta = chunk["meta"]

    for name, code in STATE_NAMES.items():
        if name.lower() in q:
            if meta.get("state_code") == code or name.lower() in text:
                break
            return False, f"вопрос про {name}, чанк к штату не относится"

    for cat in CATEGORIES:
        if cat.replace("_", " ") in q and meta.get("category") not in (None, cat):
            if cat.replace("_", " ") not in text:
                return False, f"вопрос про {cat}, чанк к категории не относится"
    return True, ""


def clean(path) -> tuple[int, int]:
    by_id = {c["chunk_id"]: c for c in load_chunks()}
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    kept, dropped = [], []
    for row in rows:
        ok, why = entity_supported(row["question"], by_id[row["gold_chunk_id"]])
        (kept if ok else dropped).append((row, why))

    with path.open("w", encoding="utf-8") as f:
        for row, _ in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"  {path.name}: было {len(rows)}, осталось {len(kept)}, отброшено {len(dropped)}")
    for row, why in dropped:
        print(f"     - {row['question'][:62]:<62} {why}")
    return len(rows), len(kept)


def main() -> None:
    for p in (GOLDEN_PATH, GOLDEN_HARD_PATH):
        clean(p)


if __name__ == "__main__":
    main()
