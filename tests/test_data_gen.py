# Requirements traceability for this file:
#   FR-2  customer attributes / login / usage / payments / support counts as inputs
#   FR-20 reproducible synthetic data from a fixed seed + generation script
#   FR-21 class imbalance (churn is the minority class)
#   NFR-2 synthetic Telco-like data + >=100 support texts, offline
#
# covers: FR-2
# covers: FR-20
# covers: FR-21
# covers: NFR-2
from __future__ import annotations

import pandas as pd

from churn_recommend import config, data_gen


def test_generate_all_shapes_and_inputs():
    # covers: FR-2  -- required input columns are present
    tables = data_gen.generate_all(n_customers=120, n_texts=130, seed=config.SEED)
    customers = tables["customers"]
    assert len(customers) == 120
    for col in [
        "tenure_months",
        "monthly_charges",
        "login_freq_per_week",
        "feature_usage_score",
        "payment_failures",
        "support_calls",
        "contract_type",
        "payment_method",
        "churned",
    ]:
        assert col in customers.columns


def test_support_texts_count_and_linkage():
    # covers: NFR-2  -- >=100 support texts, each linked to a real customer_id
    tables = data_gen.generate_all(seed=config.SEED)
    support = tables["support_texts"]
    customers = tables["customers"]
    assert len(support) >= 100
    assert set(support["customer_id"]).issubset(set(customers["customer_id"]))
    # cancellation-hint phrases must appear in the corpus
    joined = " ".join(support["text"].tolist())
    assert "解約" in joined
    assert (support["sentiment"] == "cancellation").any()


def test_class_imbalance_minority_churn():
    # covers: FR-21  -- churners are a minority class
    tables = data_gen.generate_all(seed=config.SEED)
    rate = tables["customers"]["churned"].mean()
    assert 0.0 < rate < 0.5


def test_reproducible_seed():
    # covers: FR-20  -- identical output for the same seed
    a = data_gen.generate_all(seed=123)
    b = data_gen.generate_all(seed=123)
    pd.testing.assert_frame_equal(a["customers"], b["customers"])
    pd.testing.assert_frame_equal(a["support_texts"], b["support_texts"])


def test_write_and_load_offline(tmp_path):
    # covers: NFR-2  -- data persists to CSV and reloads with no network
    # Write to a tmp dir so the test never clobbers the committed demo sample.
    tables = data_gen.generate_all(n_customers=50, n_texts=120, seed=1)
    data_gen.write_csvs(tables, tmp_path)
    loaded = data_gen.load_csvs(tmp_path)
    assert len(loaded["customers"]) == len(tables["customers"])
