"""Cleaning the golden set of questions the gold chunk does not answer.

A defect found while testing a hypothesis: the prompt required naming a state or
a category for the sake of unambiguity, and the model named them even where the
chunk had nothing to do with any state or category. The result was a question
like "when will my stationery order arrive in Sao Paulo" whose assigned gold
chunk is a category handbook that never mentions Sao Paulo at all.

Such a question is unanswerable as posed, and counting it as a retrieval failure
is lying to yourself.
"""
from __future__ import annotations

import json

from src.eval.golden import GOLDEN_HARD_PATH, GOLDEN_PATH
from src.ingest.build_docs import CATEGORIES, STATES
from src.rag.index import load_chunks

STATE_NAMES = {n: c for c, n, _ in STATES}


def entity_supported(question: str, chunk: dict) -> tuple[bool, str]:
    """If a question names a state or a category, the chunk must support it."""
    q = question.lower()
    text = chunk["text"].lower()
    meta = chunk["meta"]

    for name, code in STATE_NAMES.items():
        if name.lower() in q:
            if meta.get("state_code") == code or name.lower() in text:
                break
            return False, f"question about {name}, chunk unrelated to that state"

    for cat in CATEGORIES:
        if cat.replace("_", " ") in q and meta.get("category") not in (None, cat):
            if cat.replace("_", " ") not in text:
                return False, f"question about {cat}, chunk unrelated to that category"
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

    print(f"  {path.name}: was {len(rows)}, kept {len(kept)}, dropped {len(dropped)}")
    for row, why in dropped:
        print(f"     - {row['question'][:62]:<62} {why}")
    return len(rows), len(kept)


def main() -> None:
    for p in (GOLDEN_PATH, GOLDEN_HARD_PATH):
        clean(p)


if __name__ == "__main__":
    main()
