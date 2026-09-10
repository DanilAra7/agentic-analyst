"""Golden set for the SQL tool.

The reference is the RESULT, not the text of the query (decision #17). One
question has dozens of correct queries; comparing them as strings means
penalising correct answers. The reference SQL is stored alongside, not for
comparison but so that failures can be analysed.

The questions are written by hand. Synthetic generation is more dangerous here
than in RAG: the model would invent questions it already knows how to answer, and
the measurement would be inflated by construction.

The `trap` field ties a question to a class of trap, so that failures can be
counted per class rather than as one number.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from src.config import EVALS

GOLDEN_SQL_PATH = EVALS / "golden_sql.jsonl"


@dataclass
class SqlQuestion:
    id: str
    question: str
    reference_sql: str
    trap: str = "-"            # trap class code, or "-" for the plain ones
    note: str = ""             # how exactly the model is expected to go wrong
    unanswerable: bool = False # the correct behaviour is a refusal, not a number
    wrong_sql: str = ""        # the PREDICTED wrong query, see below
    expected: list = field(default_factory=list)
    wrong_expected: list = field(default_factory=list)
    columns: list = field(default_factory=list)

# Why `wrong_sql` exists. Without it a failure is just "did not match", and we do
# not know whether the model got the grain wrong or mixed up a filter. With it a
# failure is classified: if the answer matched the predicted wrong number, that is
# a GRAIN error, and its share is exactly what should fall as the schema
# description level rises. Otherwise the measurement would say "it got better"
# without saying what got better.


Q = [
    # ---------- plain: check that the basics work at all ----------
    SqlQuestion("s01", "How many orders are there in total?",
                "SELECT COUNT(*) AS n FROM order_summary"),
    SqlQuestion("s02", "How many orders have the status 'delivered'?",
                "SELECT COUNT(*) AS n FROM order_summary WHERE order_status='delivered'"),
    SqlQuestion("s03", "How many distinct product categories are there?",
                "SELECT COUNT(DISTINCT category) AS n FROM order_facts"),
    SqlQuestion("s04", "What is the total revenue from item prices, excluding freight?",
                "SELECT SUM(price) AS revenue FROM order_facts"),
    SqlQuestion("s05", "What is the average freight value per item?",
                "SELECT AVG(freight_value) AS avg_freight FROM order_facts"),
    SqlQuestion("s06", "Which customer state has the most orders? Return the state and the count.",
                "SELECT customer_state, COUNT(*) AS n FROM order_summary "
                "GROUP BY 1 ORDER BY n DESC LIMIT 1"),
    SqlQuestion("s07", "How many orders were placed in 2017?",
                "SELECT COUNT(*) AS n FROM order_summary "
                "WHERE purchased_at >= '2017-01-01' AND purchased_at < '2018-01-01'"),
    SqlQuestion("s08", "What is the highest single item price?",
                "SELECT MAX(price) AS max_price FROM order_facts"),
    SqlQuestion("s09", "How many orders were canceled?",
                "SELECT COUNT(*) AS n FROM order_summary WHERE order_status='canceled'"),
    SqlQuestion("s10", "List the top 3 categories by number of items sold, with the counts.",
                "SELECT category, COUNT(*) AS items FROM order_facts "
                "WHERE category IS NOT NULL GROUP BY 1 ORDER BY items DESC LIMIT 3"),
    SqlQuestion("s11", "What is the average number of items per order?",
                "SELECT AVG(item_count) AS avg_items FROM order_summary"),
    SqlQuestion("s12", "How many orders came from the state of Rio de Janeiro (RJ)?",
                "SELECT COUNT(*) AS n FROM order_summary WHERE customer_state='RJ'"),

    # ---------- A. grain ----------
    SqlQuestion("a01", "How many orders were placed by customers in the state of Sao Paulo (SP)?",
                "SELECT COUNT(*) AS n FROM order_summary WHERE customer_state='SP'",
                trap="A1", note="COUNT(*) over order_facts counts items, not orders",
                wrong_sql="SELECT COUNT(*) AS n FROM order_facts WHERE customer_state='SP'"),
    SqlQuestion("a02", "What is the average total value of an order, counting only item prices?",
                "SELECT AVG(items_total) AS avg_order_value FROM order_summary",
                trap="A2", note="AVG(price) over items gives the mean item price, not the basket",
                wrong_sql="SELECT AVG(price) AS v FROM order_facts"),
    SqlQuestion("a03", "What is the average price of a single item?",
                "SELECT AVG(price) AS avg_item_price FROM order_facts",
                trap="A2", note="the mirror of a02: here the item-grain mart is the right one"),
    SqlQuestion("a04", "How many distinct orders contain at least one item "
                       "from the 'bed_bath_table' category?",
                "SELECT COUNT(DISTINCT order_id) AS n FROM order_facts "
                "WHERE category='bed_bath_table'",
                trap="A1", note="COUNT(*) counts items: an order with 3 such products gives 3",
                wrong_sql="SELECT COUNT(*) AS n FROM order_facts WHERE category='bed_bath_table'"),
    SqlQuestion("a05", "What is the average review score of orders that contain at least one "
                       "item from the 'bed_bath_table' category? Count each order once.",
                "SELECT AVG(s.review_score) AS avg_score FROM order_summary s "
                "WHERE s.order_id IN (SELECT order_id FROM order_facts "
                "WHERE category='bed_bath_table')",
                trap="A3", note="mixed grain: a naive JOIN counts an order as many times "
                                "as it has such items",
                wrong_sql="SELECT AVG(s.review_score) AS v FROM order_facts f "
                          "JOIN order_summary s USING(order_id) "
                          "WHERE f.category='bed_bath_table'"),
    SqlQuestion("a06", "What is the total revenue including freight?",
                "SELECT SUM(price+freight_value) AS revenue FROM order_facts",
                trap="A1", note="SUM(items_total+freight_total) over orders gives the same; "
                                "the error appears when the marts are mixed"),

    # ---------- B. NULL and completeness ----------
    SqlQuestion("b01", "For orders that were actually delivered, what is the average number of "
                       "days between purchase and delivery?",
                "SELECT AVG(date_diff('day', purchased_at, delivered_at)) AS avg_days "
                "FROM order_summary WHERE delivered_at IS NOT NULL",
                trap="B1", note="2,965 orders have no delivery date; the error is in the denominator",
                wrong_sql="SELECT ROUND(SUM(date_diff('day', purchased_at, delivered_at))"
                          "/COUNT(*),4) AS v FROM order_summary"),
    SqlQuestion("b02", "How many orders have no delivery date recorded?",
                "SELECT COUNT(*) AS n FROM order_summary WHERE delivered_at IS NULL",
                trap="B1"),
    SqlQuestion("b03", "How many orders have status 'delivered' but no delivery date recorded?",
                "SELECT COUNT(*) AS n FROM order_summary "
                "WHERE order_status='delivered' AND delivered_at IS NULL",
                trap="B2", note="the data contradicts itself"),
    SqlQuestion("b04", "Among orders where the delivery delay is known, what percentage arrived "
                       "later than the estimated date? Return the percentage.",
                "SELECT ROUND(100.0*COUNT(*) FILTER (WHERE delivery_delay_days>0)"
                "/COUNT(delivery_delay_days),4) AS pct FROM order_summary",
                trap="B3", note="the denominator: all orders, or only the measurable ones",
                wrong_sql="SELECT ROUND(100.0*COUNT(*) FILTER (WHERE delivery_delay_days>0)"
                          "/COUNT(*),4) AS pct FROM order_summary"),
    SqlQuestion("b05", "How many orders have no review score?",
                "SELECT COUNT(*) AS n FROM order_summary WHERE review_score IS NULL",
                trap="B1"),

    # ---------- C. filters and boundaries ----------
    SqlQuestion("c01", "How many orders were placed in the fourth quarter of 2018?",
                "SELECT COUNT(*) AS n FROM order_summary "
                "WHERE purchased_at >= '2018-10-01' AND purchased_at < '2019-01-01'",
                trap="C2", note="the data stops on 2018-10-17, the quarter is incomplete"),
    SqlQuestion("c02", "How many orders were placed in 2019?",
                "SELECT COUNT(*) AS n FROM order_summary "
                "WHERE purchased_at >= '2019-01-01' AND purchased_at < '2020-01-01'",
                trap="C2", note="zero is not an answer on the merits: there is no 2019 data at all"),
    SqlQuestion("c03", "How many orders arrived earlier than the estimated delivery date?",
                "SELECT COUNT(*) AS n FROM order_summary WHERE delivery_delay_days < 0",
                trap="C3", note="the sign: a positive value means late",
                wrong_sql="SELECT COUNT(*) AS n FROM order_summary WHERE delivery_delay_days > 0"),
    SqlQuestion("c04", "How many orders arrived later than the estimated delivery date?",
                "SELECT COUNT(*) AS n FROM order_summary WHERE delivery_delay_days > 0",
                trap="C3"),
    SqlQuestion("c05", "How many orders were placed in the first half of 2018, "
                       "between January 1 and June 30 inclusive?",
                "SELECT COUNT(*) AS n FROM order_summary "
                "WHERE purchased_at >= '2018-01-01' AND purchased_at < '2018-07-01'",
                trap="C2", note="the boundaries are inclusive"),

    # ---------- D. entity identity ----------
    SqlQuestion("d01", "How many unique customers placed at least one order?",
                "SELECT COUNT(DISTINCT customer_id) AS n FROM order_summary",
                trap="D1", unanswerable=True,
                note="customer_id is unique per order; the real customer_unique_id is "
                     "absent from the marts. Any answer equals the order count and is "
                     "wrong. The correct behaviour is to say the data is not there"),

    # ---------- F. phantom rows from a LEFT JOIN ----------
    SqlQuestion("f01", "How many individual items were sold in total across all orders?",
                "SELECT COUNT(order_item_id) AS n FROM order_facts",
                trap="F1", note="775 orders contain no items at all, but the LEFT JOIN left "
                                "them in the mart as an empty row: COUNT(*) gives 113,425 "
                                "instead of 112,650",
                wrong_sql="SELECT COUNT(*) AS n FROM order_facts"),
    SqlQuestion("f02", "How many orders contain no items at all?",
                "SELECT COUNT(*) AS n FROM order_facts WHERE order_item_id IS NULL",
                trap="F1", note="the very existence of such orders is not obvious"),

    # ---------- E. ranking ----------
    SqlQuestion("e01", "List the top 3 categories by total revenue, counting item prices only. "
                       "Return category and revenue.",
                "SELECT category, SUM(price) AS revenue FROM order_facts "
                "WHERE category IS NOT NULL GROUP BY 1 ORDER BY revenue DESC LIMIT 3",
                trap="E1", note="revenue without freight"),
    SqlQuestion("e02", "List the top 3 categories by total revenue including freight. "
                       "Return category and revenue.",
                "SELECT category, SUM(price+freight_value) AS revenue FROM order_facts "
                "WHERE category IS NOT NULL GROUP BY 1 ORDER BY revenue DESC LIMIT 3",
                trap="E1", note="revenue with freight; the totals differ by up to 20%"),
    SqlQuestion("e03", "Which 3 states have the lowest average review score? "
                       "Return state and average score.",
                "SELECT customer_state, AVG(review_score) AS avg_score "
                "FROM order_summary WHERE review_score IS NOT NULL "
                "GROUP BY 1 ORDER BY avg_score ASC LIMIT 3",
                trap="E1", note="NULLs in the scores must be excluded explicitly"),
]


def build() -> None:
    from src.tools.sql import run

    out, bad = [], []
    for q in Q:
        r = run(q.reference_sql)
        if not r.ok:
            bad.append((q.id, r.error))
            continue
        q.columns = r.columns
        q.expected = [list(row) for row in r.rows]
        if q.wrong_sql:
            w = run(q.wrong_sql)
            if not w.ok:
                bad.append((q.id + " (wrong_sql)", w.error)); continue
            q.wrong_expected = [list(row) for row in w.rows]
            if q.wrong_expected == q.expected:
                bad.append((q.id, "the predicted wrong query returned THE SAME result: "
                                  "the trap does not catch anything")); continue
        out.append(q)

    if bad:
        print("REFERENCE QUERIES FAILED TO RUN:")
        for i, e in bad:
            print(f"  {i}: {e}")
        raise SystemExit(1)

    with GOLDEN_SQL_PATH.open("w", encoding="utf-8") as f:
        for q in out:
            f.write(json.dumps(asdict(q), ensure_ascii=False, default=str) + "\n")

    print(f"questions: {len(out)}   traps: {sum(1 for q in out if q.trap != '-')}"
          f"   unanswerable: {sum(1 for q in out if q.unanswerable)}")
    print(f"{GOLDEN_SQL_PATH}\n")
    print(f"{'id':<5}{'trap':<6}{'question':<58}{'reference':<26}predicted error")
    print("-" * 130)
    for q in out:
        val = "; ".join(", ".join(str(x) for x in row) for row in q.expected[:2]) or "(empty)"
        wrong = ("; ".join(", ".join(str(x) for x in row) for row in q.wrong_expected[:1])
                 if q.wrong_expected else "")
        print(f"{q.id:<5}{q.trap:<6}{q.question[:56]:<58}{val[:24]:<26}{wrong[:22]}")


if __name__ == "__main__":
    build()
