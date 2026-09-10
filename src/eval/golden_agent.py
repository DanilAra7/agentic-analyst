"""Golden set for the agent: four kinds of question.

Why four kinds rather than one common set. The router (one call: pick a tool and
run it) and the agent (a loop) differ EXACTLY on the two-source questions: the
router physically makes one trip to one tool. Without separating the kinds, the
difference is averaged away and becomes invisible - precisely the mistake we
already caught in M2 with the reranker.

  sql    answerable only from numbers in the database
  docs   answerable only from the text of the policies
  both   requires BOTH sources: a fact from a document + a computation over the
         database
  none   not answerable at all, the correct behaviour is a refusal

The test for a genuine `both`: could a person answer having only one tool? If yes,
the question is defective. Every both-question below is built so that the number
from the document (a threshold, a window, a list of categories) cannot be obtained
from the database, and the count from the database cannot be obtained from the
document.

`needs_tools` is labelled BY HAND. It is the reference for the "did the agent pick
the right tool" metric - the only one that does not depend on whether the judge
lies (decision #19).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from src.config import EVALS

GOLDEN_AGENT_PATH = EVALS / "golden_agent.jsonl"
SQL, DOCS = "sql_query", "search_docs"


@dataclass
class AgentQuestion:
    id: str
    kind: str                    # sql | docs | both | none
    question: str
    needs_tools: list[str]
    expected_value: float | None = None   # if the answer is a number, matched exactly
    expected_facts: list[str] = field(default_factory=list)  # what must appear in the answer
    reference_sql: str = ""      # how the number was computed, for debugging failures
    doc_fact: str = ""           # which fact is taken from the document
    note: str = ""


Q = [
    # ---------------- SQL only ----------------
    AgentQuestion("sq1", "sql", "How many orders have the status 'delivered'?", [SQL],
                  96478, reference_sql="SELECT COUNT(*) FROM order_summary WHERE order_status='delivered'"),
    AgentQuestion("sq2", "sql", "Which customer state has the most orders?", [SQL],
                  expected_facts=["SP"],
                  reference_sql="SELECT customer_state FROM order_summary GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 1"),
    AgentQuestion("sq3", "sql", "What is the average review score across all orders?", [SQL],
                  4.0868, reference_sql="SELECT AVG(review_score) FROM order_summary"),
    AgentQuestion("sq4", "sql", "Which product category has the most items sold?", [SQL],
                  expected_facts=["bed_bath_table"],
                  reference_sql="SELECT category FROM order_facts WHERE category IS NOT NULL "
                                "GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 1"),
    AgentQuestion("sq5", "sql", "How many orders arrived later than the estimated delivery date?", [SQL],
                  6535, reference_sql="SELECT COUNT(*) FROM order_summary WHERE delivery_delay_days>0"),
    AgentQuestion("sq6", "sql", "What is the total revenue including freight?", [SQL],
                  15843553.24,
                  reference_sql="SELECT SUM(price+freight_value) FROM order_facts"),

    # ---------------- documents only ----------------
    AgentQuestion("dq1", "docs", "What is the voluntary return window under the current returns policy?",
                  [DOCS], expected_facts=["14"],
                  doc_fact="POL-RET-002 section 2: 14 calendar days"),
    AgentQuestion("dq2", "docs", "What is the return window for the telephony category?", [DOCS],
                  expected_facts=["30"], doc_fact="category-restrictions: telephony = 30 days"),
    AgentQuestion("dq3", "docs", "Which categories require prior authorisation for a return?", [DOCS],
                  expected_facts=["auto", "electronics", "musical_instruments"],
                  doc_fact="category-restrictions, column 'Prior authorisation' = Yes"),
    AgentQuestion("dq4", "docs", "A delivery is 5 days beyond the committed deadline. "
                                 "What compensation is due?", [DOCS],
                  expected_facts=["freight", "10%"],
                  doc_fact="delay compensation, tier 4-7 days: freight refunded plus 10% of item"),
    AgentQuestion("dq5", "docs", "Does the current policy charge a restocking fee "
                                 "on defect-related returns?", [DOCS],
                  expected_facts=["no"], doc_fact="POL-RET-002 section 3: abolished for defect returns",
                  note="a trap: the superseded v1 charges 15%, the current one does not"),
    AgentQuestion("dq6", "docs", "Is Saturday counted as a business day for parcels "
                                 "handled through the Roraima hub?", [DOCS],
                  expected_facts=["not"], doc_fact="OPS-RR-001 section 2: Saturday is NOT a business day"),

    # ---------------- both sources ----------------
    AgentQuestion("bq1", "both",
                  "Under the delivery delay compensation policy, how many orders qualify "
                  "for the highest compensation tier?", [SQL, DOCS], 1384,
                  reference_sql="SELECT COUNT(*) FROM order_summary WHERE delivery_delay_days>=15",
                  doc_fact="the highest tier threshold = 15 days or more",
                  note="the threshold 15 cannot come from the database, the count cannot come "
                       "from the document"),
    AgentQuestion("bq2", "both",
                  "How much freight would we refund in total for orders that fall into the "
                  "lowest delay compensation tier?", [SQL, DOCS], 44098.66,
                  reference_sql="SELECT SUM(freight_total) FROM order_summary "
                                "WHERE delivery_delay_days BETWEEN 1 AND 3",
                  doc_fact="lowest tier = 1-3 days late, freight refunded in full"),
    AgentQuestion("bq3", "both",
                  "Among the three categories with the highest revenue, which one has the "
                  "shortest return window?", [SQL, DOCS], expected_facts=["watches_gifts", "7"],
                  reference_sql="SELECT category FROM order_facts WHERE category IS NOT NULL "
                                "GROUP BY 1 ORDER BY SUM(price) DESC LIMIT 3",
                  doc_fact="return windows: health_beauty 14, watches_gifts 7, bed_bath_table 14"),
    AgentQuestion("bq4", "both",
                  "For the state with the most orders, what is the designated sorting hub?",
                  [SQL, DOCS], expected_facts=["São Paulo"],
                  reference_sql="SELECT customer_state FROM order_summary GROUP BY 1 "
                                "ORDER BY COUNT(*) DESC LIMIT 1",
                  doc_fact="OPS-SP-001: the hub for SP is São Paulo"),
    AgentQuestion("bq5", "both",
                  "How many distinct orders contain items from categories that require "
                  "prior authorisation for returns?", [SQL, DOCS], 7073,
                  reference_sql="SELECT COUNT(DISTINCT order_id) FROM order_facts WHERE category "
                                "IN ('auto','electronics','musical_instruments')",
                  doc_fact="the list of categories requiring prior authorisation"),
    AgentQuestion("bq6", "both",
                  "The state with the worst average review score: is Saturday counted as a "
                  "business day for its hub?", [SQL, DOCS], expected_facts=["not"],
                  reference_sql="SELECT customer_state FROM order_summary WHERE review_score "
                                "IS NOT NULL GROUP BY 1 ORDER BY AVG(review_score) LIMIT 1",
                  doc_fact="the worst state is RR; OPS-RR-001: Saturday is NOT a business day"),
    AgentQuestion("bq7", "both",
                  "Among the five categories with the most items sold, which has the "
                  "shortest return window?", [SQL, DOCS], expected_facts=["sports_leisure", "7"],
                  reference_sql="SELECT category FROM order_facts WHERE category IS NOT NULL "
                                "GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 5",
                  doc_fact="sports_leisure = 7 days, the rest of the top 5 = 14"),
    AgentQuestion("bq8", "both",
                  "Which of the top five categories by items sold has a hygiene seal "
                  "requirement on returns?", [SQL, DOCS], expected_facts=["health_beauty"],
                  reference_sql="SELECT category FROM order_facts WHERE category IS NOT NULL "
                                "GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 5",
                  doc_fact="hygiene seal = Yes only for health_beauty, baby, perfumery"),

    # ---------------- not answerable ----------------
    AgentQuestion("nq1", "none", "How many unique customers placed at least one order?", [],
                  note="customer_id is unique per order, the real customer_unique_id is absent. "
                       "Any answer equals the order count and is wrong"),
    AgentQuestion("nq2", "none", "How many orders were paid by credit card?", [],
                  note="the payments table is deliberately not exposed in the marts "
                       "(decision #16, backlog #28). Our own restriction makes the question "
                       "unanswerable"),
    AgentQuestion("nq3", "none", "What is the average delivery time to Portugal?", [],
                  note="the data covers Brazil only; international shipping is in neither the "
                       "database nor the policies"),
    AgentQuestion("nq4", "none", "What is the contact email of the seller with the highest revenue?", [],
                  note="the seller can be computed, the contact details exist nowhere. A partly "
                       "answerable question: the correct behaviour is to name the seller and say "
                       "the contact details are missing"),
]


def build() -> None:
    from src.tools.sql import run

    bad = []
    for q in Q:
        if q.reference_sql:
            r = run(q.reference_sql)
            if not r.ok:
                bad.append((q.id, r.error))
    if bad:
        print("REFERENCE QUERIES FAILED TO RUN:")
        for i, e in bad:
            print(f"  {i}: {e}")
        raise SystemExit(1)

    with GOLDEN_AGENT_PATH.open("w", encoding="utf-8") as f:
        for q in Q:
            f.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")

    from collections import Counter
    c = Counter(q.kind for q in Q)
    print(f"questions: {len(Q)}   " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))
    print(f"{GOLDEN_AGENT_PATH}\n")
    print(f"{'id':<5}{'kind':<6}{'tools':<22}{'question':<64}reference")
    print("-" * 132)
    for q in Q:
        tools = ", ".join(t.replace("_query", "").replace("search_", "") for t in q.needs_tools) or "-"
        exp = (f"{q.expected_value:,.2f}" if q.expected_value is not None
               else ", ".join(q.expected_facts) or "refusal")
        print(f"{q.id:<5}{q.kind:<6}{tools:<22}{q.question[:62]:<64}{exp[:30]}")


if __name__ == "__main__":
    build()
