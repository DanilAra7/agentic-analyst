"""Получение датасета Olist.

Датасет лежит на Kaggle (olistbr/brazilian-ecommerce) и требует авторизации,
поэтому автоматическая загрузка возможна только через настроенный kaggle CLI.
Альтернатива: положить CSV в data/raw руками.
"""
from __future__ import annotations

import shutil
import subprocess
import sys

from src.config import RAW

SLUG = "olistbr/brazilian-ecommerce"

EXPECTED = [
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_products_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]


def missing() -> list[str]:
    return [f for f in EXPECTED if not (RAW / f).exists()]


def main() -> int:
    if not (gone := missing()):
        print(f"Все {len(EXPECTED)} файлов на месте: {RAW}")
        return 0

    if shutil.which("kaggle"):
        print(f"Качаю {SLUG} через kaggle CLI...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", SLUG, "-p", str(RAW), "--unzip"],
            check=True,
        )
        gone = missing()

    if gone:
        print(
            f"Не хватает {len(gone)} файлов:\n  " + "\n  ".join(gone) + "\n\n"
            f"Положи их в {RAW}\n"
            f"Источник: https://www.kaggle.com/datasets/{SLUG}\n"
            "Либо настрой kaggle CLI: pip install kaggle, затем положи "
            "kaggle.json в ~/.kaggle/ (Kaggle > Settings > Create New Token).",
            file=sys.stderr,
        )
        return 1

    print("Готово.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
