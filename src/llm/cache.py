"""SQLite cache for LLM calls.

Why: running a 200-question golden set across 6 configurations is 1200+ requests.
Free tiers cannot absorb that, and repeat runs have to be free and reproducible.
The key includes everything that can affect the answer.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_cache (
    key         TEXT PRIMARY KEY,
    provider    TEXT NOT NULL,
    model       TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def make_key(provider: str, model: str, messages: list[dict], **params: Any) -> str:
    blob = json.dumps(
        {"p": provider, "m": model, "msg": messages, "prm": params},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


class LLMCache:
    def __init__(self, path: Path):
        self.path = path
        with _LOCK, sqlite3.connect(self.path) as con:
            con.executescript(_SCHEMA)

    def get(self, key: str) -> dict | None:
        with _LOCK, sqlite3.connect(self.path) as con:
            row = con.execute(
                "SELECT payload FROM llm_cache WHERE key = ?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, provider: str, model: str, payload: dict) -> None:
        with _LOCK, sqlite3.connect(self.path) as con:
            con.execute(
                "INSERT OR REPLACE INTO llm_cache (key, provider, model, payload) "
                "VALUES (?, ?, ?, ?)",
                (key, provider, model, json.dumps(payload, ensure_ascii=False)),
            )

    def stats(self) -> dict[str, int]:
        with _LOCK, sqlite3.connect(self.path) as con:
            rows = con.execute(
                "SELECT provider || '/' || model, COUNT(*) FROM llm_cache GROUP BY 1"
            ).fetchall()
        return dict(rows)
