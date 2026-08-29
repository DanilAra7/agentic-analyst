"""Сборка DuckDB из CSV Olist.

DuckDB читает CSV напрямую, поэтому ETL здесь минимальный: типизация,
переименование в человекочитаемые имена и одна витрина для агента.
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
}

# Денормализованная витрина: агент почти всегда джойнит одно и то же,
# и заставлять модель каждый раз собирать пятитабличный JOIN значит
# множить ошибки на ровном месте.
MART = """
CREATE OR REPLACE VIEW order_facts AS
SELECT
    o.order_id,
    o.customer_id,
    o.order_status,
    o.order_purchase_timestamp                       AS purchased_at,
    o.order_delivered_customer_date                  AS delivered_at,
    o.order_estimated_delivery_date                  AS estimated_delivery_at,
    date_diff('day', o.order_estimated_delivery_date,
                     o.order_delivered_customer_date) AS delivery_delay_days,
    i.order_item_id,
    i.product_id,
    i.seller_id,
    i.price,
    i.freight_value,
    COALESCE(t.product_category_name_english,
             p.product_category_name)                AS category,
    c.customer_state,
    c.customer_city,
    r.review_score
FROM orders o
LEFT JOIN order_items i           ON i.order_id   = o.order_id
LEFT JOIN products   p            ON p.product_id = i.product_id
LEFT JOIN category_translation t  ON t.product_category_name = p.product_category_name
LEFT JOIN customers  c            ON c.customer_id = o.customer_id
LEFT JOIN reviews    r            ON r.order_id    = o.order_id;
"""


def main() -> None:
    con = duckdb.connect(str(DB_PATH))
    for name, csv in TABLES.items():
        path = RAW / csv
        if not path.exists():
            raise FileNotFoundError(f"Нет {path}. Запусти: make data")
        con.execute(
            f"CREATE OR REPLACE TABLE {name} AS "
            f"SELECT * FROM read_csv_auto('{path}', header=true)"
        )
        n = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        print(f"  {name:22s} {n:>9,} строк")

    con.execute(MART)
    con.execute("INSTALL fts; LOAD fts;")  # BM25 для гибридного поиска
    n = con.execute("SELECT COUNT(*) FROM order_facts").fetchone()[0]
    print(f"\n  order_facts (view)     {n:>9,} строк")
    print(f"\nБаза готова: {DB_PATH}")
    con.close()


if __name__ == "__main__":
    main()
