"""Разбиение корпуса на чанки.

Решение №2 в DECISIONS.md: фиксированный размер с перекрытием, без учёта
структуры документа. Это намеренно наивный бейзлайн - улучшения будут
измеряться относительно него.

Токенизируем токенизатором самой эмбеддинг-модели, а не приблизительно по
словам: «400 токенов» должно означать то же самое, что понимает модель.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from src.config import DOCS, EVALS, settings

CHUNK_TOKENS = 400
OVERLAP_TOKENS = 60
CHUNKS_PATH = EVALS / "chunks.jsonl"

# Обогащение чанка контекстом документа перед индексацией.
#   none        — как было: в индекс идёт только тело куска
#   title       — плюс строка «название документа | его код»
#   structural  — плюс последний заголовок раздела и шапка таблицы над куском
# Зачем. Кусок-продолжение уходил в индекс безымянным: название документа лежало
# в метаданных и в эмбеддинг не попадало. Вопрос «когда придёт заказ в
# Рио-Гранди-ду-Норти» не находил кусок OPS-RN-001#1, потому что этих слов в
# тексте куска нет — они остались в куске #0.
ENRICH = os.getenv("CHUNK_ENRICH", "title")

HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.M)
TABLE_HEAD = re.compile(r"^(\|[^\n]+\|)\n\|[\s:|-]+\|$", re.M)


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


def split_tokens(n_tokens: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Границы окон в ИНДЕКСАХ токенов, а не сами токены.

    Раньше функция возвращала токены, а текст чанка собирался через
    `tok.decode`. Это оказалось потерей: токенизатор bge-m3 не сохраняет
    переводы строк, и в индекс с самого M1 попадали куски со слипшимися
    строками — таблица превращалась в одну строку, заголовки теряли границы.
    Теперь длину меряем в токенах (это по-прежнему то, что понимает модель),
    а сам текст режем из ОРИГИНАЛА по символьным смещениям.
    """
    if size <= overlap:
        raise ValueError("Перекрытие должно быть меньше размера чанка")
    step = size - overlap
    out = []
    for start in range(0, max(n_tokens, 1), step):
        end = min(start + size, n_tokens)
        if start >= end and out:
            break
        out.append((start, end))
        if end >= n_tokens:
            break
    return out


def context_header(meta: dict, prefix: str, mode: str) -> str:
    """Строка контекста, которая дописывается в начало куска.

    `prefix` — тело документа ДО этого куска: из него берём последний заголовок
    раздела и последнюю шапку таблицы, то есть ровно тот контекст, который
    нарезка отрезала.

    Чем платим. Одинаковая шапка у всех кусков одного документа делает их
    похожими друг на друга. Это должно помочь найти нужный ДОКУМЕНТ и может
    помешать выбрать нужный КУСОК внутри него. Обе метрики уже считаются
    (lenient и strict), так что размен будет виден, а не предполагаем.
    """
    if mode == "none":
        return ""
    parts = [f"{meta.get('title', '')} [{meta.get('document_id', '')}]".strip()]
    if mode == "structural":
        heads = HEADING.findall(prefix)
        if heads:
            parts.append(heads[-1][1].strip())
        tabs = TABLE_HEAD.findall(prefix)
        if tabs:
            parts.append(tabs[-1].strip())
    return "\n".join(p for p in parts if p) + "\n\n"


def build_chunks(mode: str | None = None) -> list[Chunk]:
    from transformers import AutoTokenizer

    mode = mode or ENRICH
    tok = AutoTokenizer.from_pretrained(settings.embed_model)
    chunks: list[Chunk] = []

    for path in sorted(DOCS.glob("*.md")):
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        doc_id = meta.get("document_id", path.stem)
        enc = tok(body, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]

        for idx, (a, b) in enumerate(split_tokens(len(offsets), CHUNK_TOKENS, OVERLAP_TOKENS)):
            char_a, char_b = offsets[a][0], offsets[b - 1][1]
            body_slice = body[char_a:char_b]          # оригинал, переводы строк на месте
            text = context_header(meta, body[:char_a], mode) + body_slice
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{idx}",
                    doc_id=doc_id,
                    source_file=path.name,
                    chunk_index=idx,
                    n_tokens=b - a,
                    text=text,
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
    print(f"  обогащение          {ENRICH:>7}")
    print(f"\n  {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
