"""Golden set для агента: четыре типа вопросов.

Зачем четыре типа, а не один общий набор. Роутер (один вызов: выбрать инструмент
и выполнить) и агент (цикл) отличаются РОВНО на двухисточниковых вопросах:
роутер физически делает один заход в один инструмент. Если не разделить типы,
разница усреднится и станет невидимой - ровно та ошибка, которую мы уже ловили
в M2 с реранкером.

  sql    отвечается только числами из базы
  docs   отвечается только текстом регламентов
  both   требует ОБА источника: факт из документа + вычисление по базе
  none   не отвечается вовсе, верное поведение - отказ

Проверка настоящности `both`: может ли человек ответить, имея только один
инструмент? Если да - вопрос бракованный. Каждый both-вопрос ниже устроен так,
что число из документа (порог, окно, список категорий) невозможно получить из
базы, а счёт по базе невозможно получить из документа.

`needs_tools` размечен РУКАМИ. Это эталон для метрики «выбрал ли агент верный
инструмент» - единственной, которая не зависит от того, врёт судья или нет
(решение №19).
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
    expected_value: float | None = None   # если ответ - число, сверяем точно
    expected_facts: list[str] = field(default_factory=list)  # что обязано быть в ответе
    reference_sql: str = ""      # чем считалось число, для разбора провалов
    doc_fact: str = ""           # какой факт берётся из документа
    note: str = ""


Q = [
    # ---------------- только SQL ----------------
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

    # ---------------- только документы ----------------
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
                  note="ловушка: отменённая версия v1 берёт 15%, действующая - нет"),
    AgentQuestion("dq6", "docs", "Is Saturday counted as a business day for parcels "
                                 "handled through the Roraima hub?", [DOCS],
                  expected_facts=["not"], doc_fact="OPS-RR-001 section 2: Saturday is NOT a business day"),

    # ---------------- оба источника ----------------
    AgentQuestion("bq1", "both",
                  "Under the delivery delay compensation policy, how many orders qualify "
                  "for the highest compensation tier?", [SQL, DOCS], 1384,
                  reference_sql="SELECT COUNT(*) FROM order_summary WHERE delivery_delay_days>=15",
                  doc_fact="порог высшего тарифа = 15 дней и более",
                  note="порог 15 невозможно узнать из базы, счёт невозможно узнать из документа"),
    AgentQuestion("bq2", "both",
                  "How much freight would we refund in total for orders that fall into the "
                  "lowest delay compensation tier?", [SQL, DOCS], 44098.66,
                  reference_sql="SELECT SUM(freight_total) FROM order_summary "
                                "WHERE delivery_delay_days BETWEEN 1 AND 3",
                  doc_fact="нижний тариф = опоздание 1-3 дня, возвращается фрахт целиком"),
    AgentQuestion("bq3", "both",
                  "Among the three categories with the highest revenue, which one has the "
                  "shortest return window?", [SQL, DOCS], expected_facts=["watches_gifts", "7"],
                  reference_sql="SELECT category FROM order_facts WHERE category IS NOT NULL "
                                "GROUP BY 1 ORDER BY SUM(price) DESC LIMIT 3",
                  doc_fact="окна возврата: health_beauty 14, watches_gifts 7, bed_bath_table 14"),
    AgentQuestion("bq4", "both",
                  "For the state with the most orders, what is the designated sorting hub?",
                  [SQL, DOCS], expected_facts=["São Paulo"],
                  reference_sql="SELECT customer_state FROM order_summary GROUP BY 1 "
                                "ORDER BY COUNT(*) DESC LIMIT 1",
                  doc_fact="OPS-SP-001: хаб для SP - São Paulo"),
    AgentQuestion("bq5", "both",
                  "How many distinct orders contain items from categories that require "
                  "prior authorisation for returns?", [SQL, DOCS], 7073,
                  reference_sql="SELECT COUNT(DISTINCT order_id) FROM order_facts WHERE category "
                                "IN ('auto','electronics','musical_instruments')",
                  doc_fact="список категорий с предварительной авторизацией"),
    AgentQuestion("bq6", "both",
                  "The state with the worst average review score: is Saturday counted as a "
                  "business day for its hub?", [SQL, DOCS], expected_facts=["not"],
                  reference_sql="SELECT customer_state FROM order_summary WHERE review_score "
                                "IS NOT NULL GROUP BY 1 ORDER BY AVG(review_score) LIMIT 1",
                  doc_fact="худший штат RR; OPS-RR-001: суббота НЕ рабочий день"),
    AgentQuestion("bq7", "both",
                  "Among the five categories with the most items sold, which has the "
                  "shortest return window?", [SQL, DOCS], expected_facts=["sports_leisure", "7"],
                  reference_sql="SELECT category FROM order_facts WHERE category IS NOT NULL "
                                "GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 5",
                  doc_fact="sports_leisure = 7 дней, остальные из топ-5 = 14"),
    AgentQuestion("bq8", "both",
                  "Which of the top five categories by items sold has a hygiene seal "
                  "requirement on returns?", [SQL, DOCS], expected_facts=["health_beauty"],
                  reference_sql="SELECT category FROM order_facts WHERE category IS NOT NULL "
                                "GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 5",
                  doc_fact="hygiene seal = Yes только у health_beauty, baby, perfumery"),

    # ---------------- не отвечается ----------------
    AgentQuestion("nq1", "none", "How many unique customers placed at least one order?", [],
                  note="customer_id уникален на заказ, настоящий customer_unique_id отсутствует. "
                       "Любой ответ равен числу заказов и неверен"),
    AgentQuestion("nq2", "none", "How many orders were paid by credit card?", [],
                  note="таблица оплат намеренно не выведена в витрины (решение №16, бэклог №28). "
                       "Наше собственное ограничение делает вопрос неотвечаемым"),
    AgentQuestion("nq3", "none", "What is the average delivery time to Portugal?", [],
                  note="данные только по Бразилии, международной доставки нет ни в базе, ни в регламентах"),
    AgentQuestion("nq4", "none", "What is the contact email of the seller with the highest revenue?", [],
                  note="продавца посчитать можно, контактов нет нигде. Частично отвечаемый вопрос: "
                       "верное поведение - назвать продавца и сказать, что контактов нет"),
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
        print("ЭТАЛОННЫЕ ЗАПРОСЫ НЕ ВЫПОЛНИЛИСЬ:")
        for i, e in bad:
            print(f"  {i}: {e}")
        raise SystemExit(1)

    with GOLDEN_AGENT_PATH.open("w", encoding="utf-8") as f:
        for q in Q:
            f.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")

    from collections import Counter
    c = Counter(q.kind for q in Q)
    print(f"вопросов: {len(Q)}   " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))
    print(f"{GOLDEN_AGENT_PATH}\n")
    print(f"{'id':<5}{'тип':<6}{'инструменты':<22}{'вопрос':<64}эталон")
    print("-" * 132)
    for q in Q:
        tools = ", ".join(t.replace("_query", "").replace("search_", "") for t in q.needs_tools) or "—"
        exp = (f"{q.expected_value:,.2f}" if q.expected_value is not None
               else ", ".join(q.expected_facts) or "отказ")
        print(f"{q.id:<5}{q.kind:<6}{tools:<22}{q.question[:62]:<64}{exp[:30]}")


if __name__ == "__main__":
    build()
