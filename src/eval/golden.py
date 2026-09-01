"""Синтез golden set.

Размеченных данных нет, поэтому строим их сами: по чанку модель пишет вопрос,
ответ на который в этом чанке содержится. Получаются пары «вопрос -> эталонный
чанк», по которым считается recall@k.

Две ловушки, которые здесь обходятся явно:

1. НЕОДНОЗНАЧНОСТЬ. В корпусе 27 почти одинаковых региональных справочников и
   250 бюллетеней. Вопрос «какой порог бесплатной доставки?» отвечается любым из
   27 документов, и эталонная метка становится враньём. Поэтому от модели
   требуется вопрос, однозначно указывающий на конкретный документ, а ответ
   без различающей сущности отбраковывается.

2. ПЕРЕКОС ПО СЕМЕЙСТВАМ. Бюллетеней в 30 раз больше, чем политик. Без
   стратификации golden set состоял бы почти только из них, и метрика мерила бы
   поиск по бюллетеням, а не по корпусу.
"""
from __future__ import annotations

import json
import random
import re

from src.config import EVALS
from src.llm import get_llm
from src.ingest.build_docs import STATES
from src.rag.index import load_chunks

STATE_NAME = {c: n for c, n, _ in STATES}

GOLDEN_PATH = EVALS / "golden_set.jsonl"
GOLDEN_HARD_PATH = EVALS / "golden_set_hard.jsonl"
SEED = 17

# Сколько вопросов брать из каждого семейства документов
# BUL исключены намеренно: 250 почти одинаковых бюллетеней, по нескольку на
# штат с повторяющимися причинами. Вопрос по такому честно отвечается несколькими
# документами сразу, и единственная эталонная метка была бы неверной. В корпусе
# они остаются как отвлекающие документы - именно они делают поиск трудным.
QUOTA = {"POL": 60, "OPS": 30, "CAT": 20, "FAQ": 40}

PROMPT = """You are building an evaluation set for a document search system.

Below is one passage from a marketplace policy corpus. Write EXACTLY ONE \
question that a real user might ask, which this passage answers.

Hard requirements:
- The question must be answerable from this passage alone.
- The question must be SELF-CONTAINED and UNAMBIGUOUS. The corpus contains many \
near-identical documents that differ only by Brazilian state, product category or \
document version. If this passage is specific to a state, a category, a service \
code or a version, the question MUST name it explicitly, otherwise the question \
would match dozens of other documents.
- Phrase it the way a user would, not the way the document is written. Do not \
copy whole phrases from the passage.
- Never refer to "this document", "the passage", "the text" or "section N".
- Output the question only. No preamble, no quotes, no numbering.

Passage:
---
{text}
---"""

PROMPT_HARD = """You are building a HARD evaluation set for a document search system.

Below is one passage from a marketplace policy corpus. Write EXACTLY ONE question that a real, non-expert customer would type into a help search box, which this passage answers.

This set exists to test whether search survives a VOCABULARY GAP between how users speak and how policies are written. So:
- Use everyday words. Do NOT reuse the terminology of the passage. If the passage says "reimbursement", the user says "get my money back". If it says "committed delivery deadline", the user says "when it should arrive". If it says "restocking fee", the user says "do they charge me for sending it back".
- Do not quote any phrase longer than two words from the passage.
- Do not use document codes (POL-..., RC-..., PAC-STD) unless a normal customer would plausibly know them.
- Keep it answerable from this passage alone.
- It must still be UNAMBIGUOUS: the corpus has many near-identical documents differing by Brazilian state, product category or version. Name the state or category plainly, as a customer would ("in Bahia", "for perfume").
- Never refer to "this document", "the passage", "the text" or "section N".
- Output the question only.

Passage:
---
{text}
---"""

BAD_SELF_REFERENCE = re.compile(
    r"\b(this|the)\s+(document|passage|text|section|table|handbook|bulletin|entry)\b",
    re.I,
)


def family(doc_id: str) -> str:
    return doc_id.split("-")[0].upper()


def distinguishing_tokens(meta: dict) -> list[str]:
    """Сущности, которые обязаны попасть в вопрос, иначе он неоднозначен."""
    out = []
    if code := meta.get("state_code"):
        out.append(code)
        out.append(STATE_NAME.get(code, ""))
        title = meta.get("title", "")
        if "—" in title:
            out.append(title.split("—")[-1].split("(")[0].strip())
    if cat := meta.get("category"):
        out += [cat, cat.replace("_", " ")]
    if (did := meta.get("document_id", "")).startswith(("BUL", "FAQ")):
        out.append(did)
    return [t for t in out if t]


def sample_chunks(chunks: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    buckets: dict[str, list[dict]] = {}
    for c in chunks:
        buckets.setdefault(family(c["doc_id"]), []).append(c)
    picked = []
    for fam, quota in QUOTA.items():
        pool = buckets.get(fam, [])
        rng.shuffle(pool)
        picked += pool[:quota]
    rng.shuffle(picked)
    return picked


def validate(question: str, chunk: dict) -> str | None:
    """Возвращает причину отбраковки или None, если вопрос годен."""
    q = question.strip()
    if not q or len(q) < 15:
        return "слишком короткий"
    if "\n" in q or q.count("?") > 1:
        return "не один вопрос"
    if not q.endswith("?"):
        return "не вопрос"
    if BAD_SELF_REFERENCE.search(q):
        return "ссылается на документ"
    needed = distinguishing_tokens(chunk["meta"])
    if needed and not any(t.lower() in q.lower() for t in needed):
        return f"неоднозначен, нет ни одного из {needed[:2]}"
    return None


def generate(limit: int | None = None, verbose: bool = False,
             hard: bool = False) -> list[dict]:
    llm = get_llm()
    if hard:
        # Те же самые чанки, что и в лёгком наборе. Единственная переменная —
        # формулировка вопроса, поэтому разница метрик measures ровно её.
        by_id = {c['chunk_id']: c for c in load_chunks()}
        with GOLDEN_PATH.open(encoding='utf-8') as f:
            picked = [by_id[json.loads(l)['gold_chunk_id']] for l in f]
    else:
        picked = sample_chunks(load_chunks())
    prompt = PROMPT_HARD if hard else PROMPT
    if limit:
        picked = picked[:limit]

    kept, rejected = [], []
    for i, c in enumerate(picked, 1):
        r = llm.complete(
            [{"role": "user", "content": prompt.format(text=c["text"][:4000])}],
            temperature=0.4 if hard else 0.3,
            max_tokens=120,
        )
        q = r.text.strip().strip('"')
        why = validate(q, c)
        row = {
            "question": q,
            "gold_chunk_id": c["chunk_id"],
            "gold_doc_id": c["doc_id"],
            "family": family(c["doc_id"]),
            "source_file": c["source_file"],
        }
        (rejected if why else kept).append({**row, "reject_reason": why} if why else row)
        if verbose:
            mark = "ОТБРАК" if why else "  ok  "
            print(f"[{mark}] {c['doc_id']:<32} {q[:78]}")
            if why:
                print(f"          причина: {why}")
    return kept, rejected


def main(limit: int | None = None, verbose: bool = False, hard: bool = False) -> None:
    kept, rejected = generate(limit, verbose, hard)
    EVALS.mkdir(parents=True, exist_ok=True)
    path = GOLDEN_HARD_PATH if hard else GOLDEN_PATH
    with path.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    total = len(kept) + len(rejected)
    print(f"\n  сгенерировано {total}, годных {len(kept)}, отбраковано {len(rejected)}")
    if rejected:
        from collections import Counter
        for why, n in Counter(r["reject_reason"] for r in rejected).most_common():
            print(f"    {n:>3}  {why}")
    print(f"\n  {path}")


if __name__ == "__main__":
    import sys
    hard = "--hard" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    lim = int(args[0]) if args else None
    main(lim, verbose=True, hard=hard)
