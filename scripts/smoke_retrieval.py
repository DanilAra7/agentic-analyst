"""A hand-written retrieval smoke test.

Not a replacement for the golden set. Its purpose is different: to check that the
pipeline is alive and to see in advance which classes of question dense retrieval
falls down on. Every question targets a specific difficulty planted in the corpus.
"""
from __future__ import annotations

from src.rag.retrieve import get_retriever

# (question, expected document, which difficulty it probes)
CASES = [
    ("What does return reason code RC-301 mean?",
     "POL-RET-002", "exact code"),
    ("Which return code is used when the wrong item was shipped?",
     "POL-RET-002", "exact code"),
    ("What is the free freight threshold for Bahia?",
     "OPS-BA-001", "the right state out of 27 similar ones"),
    ("How many delivery attempts are made in Amazonas?",
     "OPS-AM-001", "the right state out of 27 similar ones"),
    ("What is the SEDEX-12 rate to Acre for a 2 kg parcel?",
     "POL-RATE-001", "table"),
    ("What is the current restocking fee?",
     "POL-RET-002", "two policy versions"),
    ("How long do I have to return something?",
     "POL-RET-002", "two policy versions"),
    ("When do I get my money back after returning an item?",
     "POL-FIN-001", "vocabulary gap refund/reimbursement"),
    ("Who pays the return shipping if the item arrived broken?",
     "POL-RET-002", "cross-reference"),
    ("What compensation is due if delivery is 10 days late?",
     "POL-SLA-001", "direct question"),
    ("Is a hygiene seal relevant for perfumery products?",
     "POL-CAT-001", "category rule"),
    ("What packaging is required for musical instruments?",
     "CAT-MUSICAL-INSTRUMENTS-001", "category handbook"),
]


def main() -> None:
    r = get_retriever()
    hits_at = {1: 0, 3: 0, 5: 0, 20: 0}

    print(f"{'difficulty':<38} {'@1':>3} {'@3':>3} {'@5':>3}  question")
    print("-" * 110)
    for q, expected, kind in CASES:
        got = [h.chunk_id.split("#")[0] for h in r.search(q, k=20)]
        rank = got.index(expected) + 1 if expected in got else None
        marks = []
        for k in (1, 3, 5):
            ok = rank is not None and rank <= k
            hits_at[k] += ok
            marks.append(" +" if ok else " ·")
        if rank is not None and rank <= 20:
            hits_at[20] += 1
        pos = f"#{rank}" if rank else "none"
        print(f"{kind:<38} {marks[0]:>3}{marks[1]:>3}{marks[2]:>3}  {q[:52]:<52} {pos}")

    n = len(CASES)
    print("-" * 110)
    for k in (1, 3, 5, 20):
        print(f"  recall@{k:<3} {hits_at[k]}/{n}  = {hits_at[k]/n:.2f}")


if __name__ == "__main__":
    main()
