"""Auditing the gold labels.

Debt 16: every question has one correct label, but several documents in the
corpus may honestly answer the same question. In that case a "miss" is a labelling
defect rather than a retrieval one, and every metric is systematically understated.

Here we do NOT fix the labels; we measure the scale of the problem: for every miss
we ask the judge whether the answer is contained in what retrieval actually returned.
Cheap: misses only, top-3 only, everything cached.
"""
from __future__ import annotations

import json
import re

from src.config import EVALS
from src.eval.retrieval import load_golden
from src.llm import get_judge

TOP_N = 3

PROMPT = """You judge whether a retrieved passage answers a support agent's question.

Question:
{question}

Passage:
---
{passage}
---

Judge how well the passage answers the question. Be practical, not pedantic: the
passage does NOT need to cover every nuance or edge case. It is enough if an
agent reading it could give the customer a correct answer to what was actually
asked.

Reply in exactly two lines:
REASON: <one short sentence>
VERDICT: <FULL if the passage answers the question | PARTIAL if it gives part of
the answer but a key number or condition is missing | NO if it does not address
the question at all>"""

VERDICT_RE = re.compile(r"VERDICT:\s*(FULL|PARTIAL|NO)", re.I)


def judge_passage(judge, question: str, passage: str) -> str:
    """Returns FULL / PARTIAL / NO."""
    r = judge.complete(
        [{"role": "user", "content": PROMPT.format(question=question, passage=passage[:2500])}],
        temperature=0.0, max_tokens=120)
    m = VERDICT_RE.search(r.text or "")
    return m.group(1).upper() if m else "NO"


def main() -> None:
    from src.rag.retrieve import get_retriever

    golden = load_golden(hard=True)
    r = get_retriever()
    judge = get_judge()
    ranked = r.search_batch([g["question"] for g in golden], k=5)

    misses = [(g, rk) for g, rk in zip(golden, ranked) if g["gold_chunk_id"] not in rk]
    print(f"misses on the hard set: {len(misses)} of {len(golden)}\n")

    false_misses, true_misses, details = 0, 0, []
    for g, rk in misses:
        verdicts = []
        for cid in rk[:TOP_N]:
            passage = r.by_id[cid]["text"][:2500]
            v = judge_passage(judge, g["question"], passage)
            verdicts.append((cid, v == "FULL"))
        answered = [cid for cid, ok in verdicts if ok]
        if answered:
            false_misses += 1
            mark, note = "FALSE", f"answered by {answered[0].split('#')[0]}"
        else:
            true_misses += 1
            mark, note = "real", f"gold {g['gold_doc_id']}"
        details.append({"question": g["question"], "gold": g["gold_doc_id"],
                        "false_miss": bool(answered), "answered_by": answered})
        print(f"  [{mark:>9}] {g['question'][:66]:<66} {note}")

    n = len(misses)
    print(f"\n  false misses      {false_misses:>3} of {n}  ({false_misses/n:.0%})")
    print(f"  real misses       {true_misses:>3} of {n}")
    corrected = (len(golden) - true_misses) / len(golden)
    print(f"\n  recall@5 by label          {(len(golden)-n)/len(golden):.3f}")
    print(f"  recall@5 after the audit   {corrected:.3f}   (+{corrected-(len(golden)-n)/len(golden):.3f})")

    (EVALS / "label_audit.json").write_text(
        json.dumps({"n_misses": n, "false_misses": false_misses,
                    "details": details}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
