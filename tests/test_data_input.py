"""UI data-input path: users can upload their own customer data or use the sample.

Covers the validation/read helpers behind the sidebar data-source selector
(サンプル / アップロード). Streamlit widgets are exercised by AppTest; here we lock
the deterministic parsing/validation contract.

    # covers: FR-2   accept user customer data (attributes/usage/payments/support)
    # covers: NFR-2  bundled sample remains usable with no upload
"""
from __future__ import annotations

import io

import pytest

from churn_recommend import data_gen


def _tables() -> dict:
    return data_gen.load_csvs()


def test_sample_customers_valid_and_full_size():
    # covers: NFR-2
    customers = _tables()["customers"]
    assert data_gen.validate_customers_df(customers) == []
    # The committed sample must match the generator default so the bundled demo
    # reproduces the advertised metrics (not a degraded mini-sample).
    assert len(customers) == 400


def test_validate_reports_missing_columns():
    # covers: FR-2
    bad = _tables()["customers"].drop(columns=["churned", "support_calls"])
    assert set(data_gen.validate_customers_df(bad)) == {"churned", "support_calls"}


def test_read_customers_csv_roundtrip():
    # covers: FR-2
    buf = io.StringIO()
    _tables()["customers"].to_csv(buf, index=False)
    buf.seek(0)
    df = data_gen.read_customers_csv(buf)
    assert data_gen.validate_customers_df(df) == []


def test_read_customers_csv_raises_on_missing():
    # covers: FR-2
    bad = _tables()["customers"].drop(columns=["churned"])
    buf = io.StringIO()
    bad.to_csv(buf, index=False)
    buf.seek(0)
    with pytest.raises(ValueError, match="churned"):
        data_gen.read_customers_csv(buf)


def test_optional_purchases_and_support_readers():
    # covers: FR-2
    t = _tables()
    for reader, key in (
        (data_gen.read_purchases_csv, "purchases"),
        (data_gen.read_support_csv, "support_texts"),
    ):
        buf = io.StringIO()
        t[key].to_csv(buf, index=False)
        buf.seek(0)
        assert not reader(buf).empty
