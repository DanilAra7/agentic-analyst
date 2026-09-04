"""Golden set для SQL-инструмента.

Эталон - РЕЗУЛЬТАТ, а не текст запроса (решение №17). На один вопрос есть
десятки верных запросов; сравнивать их строками значит ругать правильные
ответы. Эталонный SQL хранится рядом, но не для сравнения, а чтобы было чем
разбирать провалы.

Вопросы написаны руками. Синтетическая генерация здесь опаснее, чем в RAG:
модель придумает вопросы, на которые сама умеет отвечать, и замер окажется
завышенным по построению.

Поле `trap` привязывает вопрос к классу ловушки, чтобы провалы можно было
считать по классам, а не одним числом.
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
    trap: str = "-"            # код класса ловушки или "-" для простых
    note: str = ""             # чем именно ошибётся модель
    unanswerable: bool = False # верное поведение - отказ, а не число
    wrong_sql: str = ""        # ПРЕДСКАЗАННЫЙ неверный запрос, см. ниже
    expected: list = field(default_factory=list)
    wrong_expected: list = field(default_factory=list)
    columns: list = field(default_factory=list)

# Зачем `wrong_sql`. Без него провал это просто «не сошлось», и мы не знаем,
# ошиблась модель зерном или перепутала фильтр. С ним провал классифицируется:
# если ответ совпал с предсказанным неверным числом, это ошибка ЗЕРНА, и
# именно её доля должна падать с ростом уровня описания схемы. Иначе замер
# показал бы «стало лучше», не говоря, что именно улучшилось.


Q = [
    # ---------- простые: проверяют, что базовое вообще работает ----------
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

    # ---------- A. зерно ----------
    SqlQuestion("a01", "How many orders were placed by customers in the state of Sao Paulo (SP)?",
                "SELECT COUNT(*) AS n FROM order_summary WHERE customer_state='SP'",
                trap="A1", note="COUNT(*) по order_facts посчитает позиции, не заказы",
                wrong_sql="SELECT COUNT(*) AS n FROM order_facts WHERE customer_state='SP'"),
    SqlQuestion("a02", "What is the average total value of an order, counting only item prices?",
                "SELECT AVG(items_total) AS avg_order_value FROM order_summary",
                trap="A2", note="AVG(price) по позициям даст среднюю цену товара, не чек",
                wrong_sql="SELECT AVG(price) AS v FROM order_facts"),
    SqlQuestion("a03", "What is the average price of a single item?",
                "SELECT AVG(price) AS avg_item_price FROM order_facts",
                trap="A2", note="зеркало a02: тут как раз нужна витрина позиций"),
    SqlQuestion("a04", "How many distinct orders contain at least one item "
                       "from the 'bed_bath_table' category?",
                "SELECT COUNT(DISTINCT order_id) AS n FROM order_facts "
                "WHERE category='bed_bath_table'",
                trap="A1", note="COUNT(*) посчитает позиции: заказ с 3 такими товарами даст 3",
                wrong_sql="SELECT COUNT(*) AS n FROM order_facts WHERE category='bed_bath_table'"),
    SqlQuestion("a05", "What is the average review score of orders that contain at least one "
                       "item from the 'bed_bath_table' category? Count each order once.",
                "SELECT AVG(s.review_score) AS avg_score FROM order_summary s "
                "WHERE s.order_id IN (SELECT order_id FROM order_facts "
                "WHERE category='bed_bath_table')",
                trap="A3", note="смешанное зерно: наивный JOIN посчитает заказ столько раз, "
                                "сколько в нём таких позиций",
                wrong_sql="SELECT AVG(s.review_score) AS v FROM order_facts f "
                          "JOIN order_summary s USING(order_id) "
                          "WHERE f.category='bed_bath_table'"),
    SqlQuestion("a06", "What is the total revenue including freight?",
                "SELECT SUM(price+freight_value) AS revenue FROM order_facts",
                trap="A1", note="SUM(items_total+freight_total) по заказам даст то же; "
                                "ошибка возникнет при смешивании витрин"),

    # ---------- B. NULL и полнота ----------
    SqlQuestion("b01", "For orders that were actually delivered, what is the average number of "
                       "days between purchase and delivery?",
                "SELECT AVG(date_diff('day', purchased_at, delivered_at)) AS avg_days "
                "FROM order_summary WHERE delivered_at IS NOT NULL",
                trap="B1", note="2 965 заказов без даты доставки; ошибка в знаменателе",
                wrong_sql="SELECT ROUND(SUM(date_diff('day', purchased_at, delivered_at))"
                          "/COUNT(*),4) AS v FROM order_summary"),
    SqlQuestion("b02", "How many orders have no delivery date recorded?",
                "SELECT COUNT(*) AS n FROM order_summary WHERE delivered_at IS NULL",
                trap="B1"),
    SqlQuestion("b03", "How many orders have status 'delivered' but no delivery date recorded?",
                "SELECT COUNT(*) AS n FROM order_summary "
                "WHERE order_status='delivered' AND delivered_at IS NULL",
                trap="B2", note="данные противоречат сами себе"),
    SqlQuestion("b04", "Among orders where the delivery delay is known, what percentage arrived "
                       "later than the estimated date? Return the percentage.",
                "SELECT ROUND(100.0*COUNT(*) FILTER (WHERE delivery_delay_days>0)"
                "/COUNT(delivery_delay_days),4) AS pct FROM order_summary",
                trap="B3", note="знаменатель: все заказы или только измеримые",
                wrong_sql="SELECT ROUND(100.0*COUNT(*) FILTER (WHERE delivery_delay_days>0)"
                          "/COUNT(*),4) AS pct FROM order_summary"),
    SqlQuestion("b05", "How many orders have no review score?",
                "SELECT COUNT(*) AS n FROM order_summary WHERE review_score IS NULL",
                trap="B1"),

    # ---------- C. фильтры и границы ----------
    SqlQuestion("c01", "How many orders were placed in the fourth quarter of 2018?",
                "SELECT COUNT(*) AS n FROM order_summary "
                "WHERE purchased_at >= '2018-10-01' AND purchased_at < '2019-01-01'",
                trap="C2", note="данные обрываются 2018-10-17, квартал неполный"),
    SqlQuestion("c02", "How many orders were placed in 2019?",
                "SELECT COUNT(*) AS n FROM order_summary "
                "WHERE purchased_at >= '2019-01-01' AND purchased_at < '2020-01-01'",
                trap="C2", note="ноль не является ответом по существу: данных за 2019 нет вовсе"),
    SqlQuestion("c03", "How many orders arrived earlier than the estimated delivery date?",
                "SELECT COUNT(*) AS n FROM order_summary WHERE delivery_delay_days < 0",
                trap="C3", note="знак: положительное значение означает опоздание",
                wrong_sql="SELECT COUNT(*) AS n FROM order_summary WHERE delivery_delay_days > 0"),
    SqlQuestion("c04", "How many orders arrived later than the estimated delivery date?",
                "SELECT COUNT(*) AS n FROM order_summary WHERE delivery_delay_days > 0",
                trap="C3"),
    SqlQuestion("c05", "How many orders were placed in the first half of 2018, "
                       "between January 1 and June 30 inclusive?",
                "SELECT COUNT(*) AS n FROM order_summary "
                "WHERE purchased_at >= '2018-01-01' AND purchased_at < '2018-07-01'",
                trap="C2", note="границы включительно"),

    # ---------- D. идентичность сущностей ----------
    SqlQuestion("d01", "How many unique customers placed at least one order?",
                "SELECT COUNT(DISTINCT customer_id) AS n FROM order_summary",
                trap="D1", unanswerable=True,
                note="customer_id уникален на заказ; настоящий customer_unique_id "
                     "в витринах отсутствует. Любой ответ равен числу заказов и неверен. "
                     "Верное поведение - сказать, что данных нет"),

    # ---------- F. фантомные строки от LEFT JOIN ----------
    SqlQuestion("f01", "How many individual items were sold in total across all orders?",
                "SELECT COUNT(order_item_id) AS n FROM order_facts",
                trap="F1", note="775 заказов не содержат позиций вовсе, но LEFT JOIN оставил "
                                "их в витрине пустой строкой: COUNT(*) даст 113 425 вместо 112 650",
                wrong_sql="SELECT COUNT(*) AS n FROM order_facts"),
    SqlQuestion("f02", "How many orders contain no items at all?",
                "SELECT COUNT(*) AS n FROM order_facts WHERE order_item_id IS NULL",
                trap="F1", note="существование таких заказов само по себе неочевидно"),

    # ---------- E. ранжирование ----------
    SqlQuestion("e01", "List the top 3 categories by total revenue, counting item prices only. "
                       "Return category and revenue.",
                "SELECT category, SUM(price) AS revenue FROM order_facts "
                "WHERE category IS NOT NULL GROUP BY 1 ORDER BY revenue DESC LIMIT 3",
                trap="E1", note="выручка без доставки"),
    SqlQuestion("e02", "List the top 3 categories by total revenue including freight. "
                       "Return category and revenue.",
                "SELECT category, SUM(price+freight_value) AS revenue FROM order_facts "
                "WHERE category IS NOT NULL GROUP BY 1 ORDER BY revenue DESC LIMIT 3",
                trap="E1", note="выручка с доставкой; суммы отличаются до 20%"),
    SqlQuestion("e03", "Which 3 states have the lowest average review score? "
                       "Return state and average score.",
                "SELECT customer_state, AVG(review_score) AS avg_score "
                "FROM order_summary WHERE review_score IS NOT NULL "
                "GROUP BY 1 ORDER BY avg_score ASC LIMIT 3",
                trap="E1", note="NULL в оценках должны быть исключены явно"),
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
                bad.append((q.id, "предсказанный неверный запрос дал ТОТ ЖЕ результат: "
                                  "ловушка не ловится")); continue
        out.append(q)

    if bad:
        print("ЭТАЛОННЫЕ ЗАПРОСЫ НЕ ВЫПОЛНИЛИСЬ:")
        for i, e in bad:
            print(f"  {i}: {e}")
        raise SystemExit(1)

    with GOLDEN_SQL_PATH.open("w", encoding="utf-8") as f:
        for q in out:
            f.write(json.dumps(asdict(q), ensure_ascii=False, default=str) + "\n")

    print(f"вопросов: {len(out)}   ловушек: {sum(1 for q in out if q.trap != '-')}"
          f"   неотвечаемых: {sum(1 for q in out if q.unanswerable)}")
    print(f"{GOLDEN_SQL_PATH}\n")
    print(f"{'id':<5}{'trap':<6}{'вопрос':<58}{'эталон':<26}предсказанная ошибка")
    print("-" * 130)
    for q in out:
        val = "; ".join(", ".join(str(x) for x in row) for row in q.expected[:2]) or "(пусто)"
        wrong = ("; ".join(", ".join(str(x) for x in row) for row in q.wrong_expected[:1])
                 if q.wrong_expected else "")
        print(f"{q.id:<5}{q.trap:<6}{q.question[:56]:<58}{val[:24]:<26}{wrong[:22]}")


if __name__ == "__main__":
    build()
