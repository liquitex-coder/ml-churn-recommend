"""Reproducible synthetic-data generation script (FR-20).

Run:
    python scripts/generate_data.py

Writes data/customers.csv, data/purchases.csv, data/support_texts.csv using a
fixed seed so output is identical on every run (FR-20, NFR-2).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from a clone without installation.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from churn_recommend import config, data_gen  # noqa: E402


def main() -> None:
    tables = data_gen.generate_all(seed=config.SEED)
    data_gen.write_csvs(tables)
    for name, df in tables.items():
        print(f"{name}: {df.shape[0]} rows x {df.shape[1]} cols -> {name}.csv")
    print(f"Data written to {config.DATA_DIR}")


if __name__ == "__main__":
    main()
