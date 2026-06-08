# Requirements traceability for this file:
#   FR-5 SHAP per-customer explanation: top contributing factors
#   FR-9 dashboard wiring -- app.py imports cleanly (3-block one-screen CRM)
#
# covers: FR-5
# covers: FR-9
from __future__ import annotations

import importlib

from churn_recommend import data_gen, text_analytics
from churn_recommend.churn_model import build_feature_frame, train_churn_model
from churn_recommend.explain import ChurnExplainer


def test_shap_top_contributors_for_customer():
    # covers: FR-5  -- per-customer top SHAP contributions are returned
    tables = data_gen.generate_all(n_customers=200, n_texts=120, seed=5)
    hint = text_analytics.compute_churn_hint_scores(
        tables["support_texts"], tables["customers"]
    )
    model = train_churn_model(tables["customers"], hint, seed=5)
    encoded = build_feature_frame(tables["customers"], hint)

    explainer = ChurnExplainer(model)
    contrib = explainer.explain_customer(encoded.loc[[0]], top_n=6)
    assert len(contrib) == 6
    for col in ["feature", "value", "shap_value", "abs_shap"]:
        assert col in contrib.columns
    # sorted by absolute contribution descending
    assert contrib["abs_shap"].is_monotonic_decreasing


def test_app_imports_without_starting_server():
    # covers: FR-9  -- dashboard module imports cleanly, no server on import
    app = importlib.import_module("app")
    assert hasattr(app, "main") and callable(app.main)
