# Requirements traceability for this file:
#   FR-1  LightGBM binary classifier -> 0-100% churn probability per customer
#   FR-4  text churn-hint score is a model feature (its contribution is visible)
#   FR-21 class imbalance handling (is_unbalance / class_weight)
#   NFR-1 runs on Python 3.10+
#   NFR-3 holdout AUC / Accuracy reported
#
# covers: FR-1
# covers: FR-4
# covers: FR-21
# covers: NFR-1
# covers: NFR-3
from __future__ import annotations

import sys

import numpy as np

from churn_recommend import config, data_gen, text_analytics
from churn_recommend.churn_model import build_feature_frame, train_churn_model


def _train(seed=11):
    tables = data_gen.generate_all(n_customers=300, n_texts=130, seed=seed)
    hint = text_analytics.compute_churn_hint_scores(
        tables["support_texts"], tables["customers"]
    )
    model = train_churn_model(tables["customers"], hint, seed=seed)
    return tables, hint, model


def test_python_version_supported():
    # covers: NFR-1  -- Python 3.10+
    assert sys.version_info >= (3, 10)


def test_probability_percent_range():
    # covers: FR-1  -- predictions are 0-100% per customer
    tables, hint, model = _train()
    encoded = build_feature_frame(tables["customers"], hint)
    pct = model.predict_proba_percent(encoded)
    assert len(pct) == len(tables["customers"])
    assert np.all((pct >= 0.0) & (pct <= 100.0))


def test_holdout_metrics_reported():
    # covers: NFR-3  -- AUC / Accuracy on holdout
    _, _, model = _train()
    assert "auc" in model.metrics and "accuracy" in model.metrics
    assert 0.5 <= model.metrics["auc"] <= 1.0
    assert 0.0 <= model.metrics["accuracy"] <= 1.0


def test_text_feature_is_in_model():
    # covers: FR-4  -- churn_hint_score participates as a model feature
    _, _, model = _train()
    assert config.TEXT_FEATURE in model.feature_columns
    imp = model.feature_importance()
    assert config.TEXT_FEATURE in imp.index


def test_class_imbalance_params_set():
    # covers: FR-21  -- imbalance handling is configured on the estimator
    _, _, model = _train()
    params = model.model.get_params()
    assert params.get("is_unbalance") is True
    assert params.get("class_weight") == "balanced"
