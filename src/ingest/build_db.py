"""Building the DuckDB database from the Olist CSVs.

DuckDB reads CSV directly, so the ETL here is minimal: typing, renaming to
human-readable names, and the marts for the agent.
"""
from __future__ import annotations

import duckdb

from src.config import DB_PATH, RAW

TABLES = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
    "geolocation": "olist_geolocation_dataset.csv",
}

# TWO marts at DIFFERENT levels of detail ("grain").
#
# This is not duplication. Joining orders with items changes the grain of the
# table: an order with three items occupies three rows, and everything that
# belongs to the order as a whole (review score, status, dates) is REPEATED.
#
# Hence the rule: an aggregate is computed at the grain where the fact actually
# lives. Price lives at the item level, the review score at the order level.
# AVG(review_score) over order_facts returns not the mean order rating but the
# mean rating weighted by basket size. The number looks plausible and is wrong -
# which is why the score was removed from the item-grain mart entirely.

MART_ITEMS = """
CREATE OR REPLACE VIEW order_facts AS
SELECT
    o.order_id,
    i.order_item_id,
    o.customer_id,
    o.order_status,
    o.order_purchase_timestamp                       AS purchased_at,
    o.order_delivered_customer_date                  AS delivered_at,
    o.order_estimated_delivery_date                  AS estimated_delivery_at,
    date_diff('day', o.order_estimated_delivery_date,
                     o.order_delivered_customer_date) AS delivery_delay_days,
    i.product_id,
    i.seller_id,
    i.price,
    i.freight_value,
    COALESCE(t.product_category_name_english,
             p.product_category_name)                AS category,
    c.customer_state,
    c.customer_city
FROM orders o
LEFT JOIN order_items i           ON i.order_id   = o.order_id
LEFT JOIN products   p            ON p.product_id = i.product_id
LEFT JOIN category_translation t  ON t.product_category_name = p.product_category_name
LEFT JOIN customers  c            ON c.customer_id = o.customer_id;
"""

MART_ORDERS = """
CREATE OR REPLACE VIEW order_summary AS
WITH items AS (
    SELECT order_id,
           COUNT(*)                AS item_count,
           SUM(price)              AS items_total,
           SUM(freight_value)      AS freight_total
    FROM order_items GROUP BY order_id
),
review AS (
    SELECT order_id, AVG(review_score) AS review_score
    FROM reviews GROUP BY order_id
)
SELECT
    o.order_id,
    o.customer_id,
    o.order_status,
    o.order_purchase_timestamp                       AS purchased_at,
    o.order_delivered_customer_date                  AS delivered_at,
    o.order_estimated_delivery_date                  AS estimated_delivery_at,
    date_diff('day', o.order_estimated_delivery_date,
                     o.order_delivered_customer_date) AS delivery_delay_days,
    COALESCE(it.item_count, 0)                       AS item_count,
    COALESCE(it.items_total, 0)                      AS items_total,
    COALESCE(it.freight_total, 0)                    AS freight_total,
    r.review_score,
    c.customer_state,
    c.customer_city
FROM orders o
LEFT JOIN items  it ON it.order_id = o.order_id
LEFT JOIN review r  ON r.order_id  = o.order_id
LEFT JOIN customers c ON c.customer_id = o.customer_id;
"""

def main() -> None:
    con = duckdb.connect(str(DB_PATH))
    for name, csv in TABLES.items():
        path = RAW / csv
        if not path.exists():
            raise FileNotFoundError(f"No {path}. Run: make data")
        con.execute(
            f"CREATE OR REPLACE TABLE {name} AS "
            f"SELECT * FROM read_csv_auto('{path}', header=true)"
        )
        n = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        print(f"  {name:22s} {n:>9,} rows")

    con.execute(MART_ITEMS)
    con.execute(MART_ORDERS)
    con.execute("INSTALL fts; LOAD fts;")  # BM25 for hybrid search
    for v in ("order_facts", "order_summary"):
        n = con.execute(f"SELECT COUNT(*) FROM {v}").fetchone()[0]
        print(f"  {v + ' (view)':22s} {n:>9,} rows")
    print(f"\nDatabase ready: {DB_PATH}")
    con.close()


if __name__ == "__main__":
    main()
