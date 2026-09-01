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
    "geolocation": "olist_geolocation_dataset.csv",
}

# ДВЕ витрины на РАЗНЫХ уровнях детализации ("зерне").
#
# Это не дублирование. Соединение заказов с позициями меняет зерно таблицы:
# заказ с тремя позициями занимает три строки, и всё, что относится к заказу
# целиком (оценка отзыва, статус, даты), в этих строках ПОВТОРЯЕТСЯ.
#
# Отсюда правило: агрегат считается на том зерне, на котором факт реально
# существует. Цена живёт на позиции, оценка отзыва - на заказе.
# AVG(review_score) по order_facts вернёт не средний рейтинг заказов, а
# средний рейтинг, взвешенный по размеру корзины. Число выглядит правдоподобно
# и является неверным - поэтому оценку из item-витрины мы убрали вовсе.

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
            raise FileNotFoundError(f"Нет {path}. Запусти: make data")
        con.execute(
            f"CREATE OR REPLACE TABLE {name} AS "
            f"SELECT * FROM read_csv_auto('{path}', header=true)"
        )
        n = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        print(f"  {name:22s} {n:>9,} строк")

    con.execute(MART_ITEMS)
    con.execute(MART_ORDERS)
    con.execute("INSTALL fts; LOAD fts;")  # BM25 для гибридного поиска
    for v in ("order_facts", "order_summary"):
        n = con.execute(f"SELECT COUNT(*) FROM {v}").fetchone()[0]
        print(f"  {v + ' (view)':22s} {n:>9,} строк")
    print(f"\nБаза готова: {DB_PATH}")
    con.close()


if __name__ == "__main__":
    main()
