"""Fetching the Olist dataset.

The dataset lives on Kaggle (olistbr/brazilian-ecommerce) and requires
authentication, so automatic download only works through a configured kaggle CLI.
The alternative is to drop the CSVs into data/raw by hand.
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
        print(f"All {len(EXPECTED)} files present: {RAW}")
        return 0

    if shutil.which("kaggle"):
        print(f"Downloading {SLUG} through the kaggle CLI...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", SLUG, "-p", str(RAW), "--unzip"],
            check=True,
        )
        gone = missing()

    if gone:
        print(
            f"Missing {len(gone)} files:\n  " + "\n  ".join(gone) + "\n\n"
            f"Put them into {RAW}\n"
            f"Source: https://www.kaggle.com/datasets/{SLUG}\n"
            "Or configure the kaggle CLI: pip install kaggle, then place "
            "kaggle.json in ~/.kaggle/ (Kaggle > Settings > Create New Token).",
            file=sys.stderr,
        )
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
