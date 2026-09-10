"""Splitting the corpus into chunks.

Decision #2 in DECISIONS.md: fixed size with overlap, ignoring document
structure. This is a deliberately naive baseline - improvements will be measured
against it.

We tokenise with the embedding model's own tokeniser rather than approximately by
words: "400 tokens" must mean the same thing the model means.
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

# Enriching a chunk with document context before indexing.
#   none        - as before: only the chunk body goes into the index
#   title       - plus a line "document title | its code"
#   structural  - plus the last section heading and table header above the chunk
# Why. A continuation chunk went into the index nameless: the document title sat
# in the metadata and never reached the embedding. The question "when will an
# order arrive in Rio Grande do Norte" did not find chunk OPS-RN-001#1, because
# those words are not in the chunk text - they stayed in chunk #0.
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
    """Minimal YAML front-matter parsing. Full YAML is not needed here: we
    generate the headers ourselves and know they hold only `key: value`.

    Parsing goes line by line rather than through `raw.startswith("---")`. The
    reason is concrete: one document out of 373 (`category-restrictions.md`) came
    out of the generator indented by 4 spaces. A start-of-string check did not
    recognise it, the front matter was not parsed, the document went into the
    index WITHOUT a title, and the header text itself ended up inside the chunk.
    The defect lived through all of M1 and M2 unnoticed, because retrieval metrics
    look only at chunk_id: the right chunk was found, and the fact that no answer
    could be extracted from it is something recall cannot see at all.
    While we are here we also strip the common indent from the body: in markdown
    4 spaces turn a table into a code block.
    """
    lines = raw.splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines) or lines[i].strip() != "---":
        return {}, raw

    meta: dict[str, str] = {}
    j = i + 1
    while j < len(lines) and lines[j].strip() != "---":
        if ":" in lines[j]:
            k, _, v = lines[j].partition(":")
            meta[k.strip()] = v.strip()
        j += 1
    body_lines = lines[j + 1:]

    body = [ln[4:] if ln.startswith("    ") else ln for ln in body_lines]
    return meta, "\n".join(body).lstrip("\n")


def split_tokens(n_tokens: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Window boundaries as token INDICES, not the tokens themselves.

    The function used to return tokens, and the chunk text was reassembled with
    `tok.decode`. That turned out to be lossy: the bge-m3 tokeniser does not
    preserve newlines, and since M1 chunks with collapsed lines had been going
    into the index - a table turned into one line, headings lost their
    boundaries. Now length is measured in tokens (still what the model
    understands), while the text itself is cut from the ORIGINAL by character
    offsets.
    """
    if size <= overlap:
        raise ValueError("Overlap must be smaller than the chunk size")
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
    """The context line prepended to the chunk.

    `prefix` is the document body BEFORE this chunk: from it we take the last
    section heading and the last table header - exactly the context that chunking
    cut away.

    What it costs. An identical header on every chunk of a document makes them
    resemble each other. That should help find the right DOCUMENT and may hurt
    picking the right CHUNK inside it. Both metrics are already computed (lenient
    and strict), so the trade-off will be visible rather than assumed.
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
            body_slice = body[char_a:char_b]          # the original, newlines intact
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

    print(f"  documents           {docs:>7,}")
    print(f"  chunks              {len(chunks):>7,}")
    print(f"  documents >1 chunk  {multi:>7,}")
    print(f"  tokens: median {sorted(toks)[len(toks)//2]:>4}  "
          f"min {min(toks):>4}  max {max(toks):>4}")
    print(f"  random recall@5     {5/len(chunks)*100:>6.2f}%")
    print(f"  enrichment          {ENRICH:>7}")
    print(f"\n  {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
