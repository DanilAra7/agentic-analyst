"""Разбиение корпуса на чанки.

Решение №2 в DECISIONS.md: фиксированный размер с перекрытием, без учёта
структуры документа. Это намеренно наивный бейзлайн - улучшения будут
измеряться относительно него.

Токенизируем токенизатором самой эмбеддинг-модели, а не приблизительно по
словам: «400 токенов» должно означать то же самое, что понимает модель.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from src.config import DOCS, EVALS, settings

CHUNK_TOKENS = 400
OVERLAP_TOKENS = 60
CHUNKS_PATH = EVALS / "chunks.jsonl"


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source_file: str
    chunk_index: int
    n_tokens: int
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


def parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    """Минимальный разбор YAML-шапки. Полный YAML тут не нужен: шапки мы
    генерируем сами и знаем, что там только `ключ: значение`."""
    if not raw.startswith("---"):
        return {}, raw
    end = raw.find("\n---", 3)
    if end == -1:
        return {}, raw
    meta = {}
    for line in raw[3:end].strip().splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    return meta, raw[end + 4 :].lstrip("\n")


def split_tokens(token_ids: list[int], size: int, overlap: int) -> list[list[int]]:
    if size <= overlap:
        raise ValueError("Перекрытие должно быть меньше размера чанка")
    step = size - overlap
    out = []
    for start in range(0, max(len(token_ids), 1), step):
        window = token_ids[start : start + size]
        if not window:
            break
        out.append(window)
        if start + size >= len(token_ids):
            break
    return out


def build_chunks() -> list[Chunk]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(settings.embed_model)
    chunks: list[Chunk] = []

    for path in sorted(DOCS.glob("*.md")):
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        doc_id = meta.get("document_id", path.stem)
        ids = tok.encode(body, add_special_tokens=False)

        for idx, window in enumerate(split_tokens(ids, CHUNK_TOKENS, OVERLAP_TOKENS)):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{idx}",
                    doc_id=doc_id,
                    source_file=path.name,
                    chunk_index=idx,
                    n_tokens=len(window),
                    text=tok.decode(window),
                    meta=meta,
                )
            )
    return chunks


def main() -> None:
    chunks = build_chunks()
    EVALS.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    docs = len({c.doc_id for c in chunks})
    toks = [c.n_tokens for c in chunks]
    multi = sum(1 for d in {c.doc_id for c in chunks}
                if sum(1 for c in chunks if c.doc_id == d) > 1)

    print(f"  документов          {docs:>7,}")
    print(f"  чанков              {len(chunks):>7,}")
    print(f"  документов >1 чанка {multi:>7,}")
    print(f"  токенов: медиана {sorted(toks)[len(toks)//2]:>4}  "
          f"мин {min(toks):>4}  макс {max(toks):>4}")
    print(f"  случайный recall@5  {5/len(chunks)*100:>6.2f}%")
    print(f"\n  {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
