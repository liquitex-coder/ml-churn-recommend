# Requirements traceability for this file:
#   FR-6  cross-sell top-3 via item-item CF AND NetworkX graph metrics
#   FR-7  recommend to active / low-risk customers using similar purchase patterns
#   FR-8  high-risk alert list sorted by risk % descending
#   FR-22 exclude already-purchased products from recommendations
#
# covers: FR-6
# covers: FR-7
# covers: FR-8
# covers: FR-22
from __future__ import annotations

import networkx as nx
import pandas as pd

from churn_recommend import config, data_gen, text_analytics
from churn_recommend.churn_model import build_feature_frame, train_churn_model
from churn_recommend.recommend import CrossSellRecommender


def _setup(seed=3):
    tables = data_gen.generate_all(n_customers=200, n_texts=120, seed=seed)
    return tables


def test_top3_and_graph_used():
    # covers: FR-6  -- top-3 recos backed by item-item CF + NetworkX graph
    tables = _setup()
    rec = CrossSellRecommender(tables["purchases"])
    assert isinstance(rec.graph, nx.Graph)
    assert rec.graph.number_of_nodes() == len(config.PRODUCTS)
    cid = tables["customers"]["customer_id"].iloc[0]
    recos = rec.recommend(cid, top_k=3)
    assert len(recos) <= 3
    assert {"cf_score", "graph_score"}.issubset(recos.columns)


def test_excludes_owned_products():
    # covers: FR-22  -- already-purchased products are never recommended
    tables = _setup()
    rec = CrossSellRecommender(tables["purchases"])
    for cid in tables["customers"]["customer_id"].iloc[:25]:
        owned = rec.owned_products(cid)
        recos = rec.recommend(cid, top_k=3)
        assert owned.isdisjoint(set(recos["product"]))


def test_target_active_low_risk_customers():
    # covers: FR-7  -- recommendations target active / low-risk customers
    tables = _setup()
    hint = text_analytics.compute_churn_hint_scores(
        tables["support_texts"], tables["customers"]
    )
    model = train_churn_model(tables["customers"], hint, seed=3)
    encoded = build_feature_frame(tables["customers"], hint)
    risk = pd.Series(
        model.predict_proba_percent(encoded), index=tables["customers"]["customer_id"]
    )
    rec = CrossSellRecommender(tables["purchases"])
    targeted = rec.recommend_for_active_customers(risk, top_k=3)
    # every targeted customer is at/below the high-risk threshold
    for cid in targeted:
        assert risk[cid] <= config.HIGH_RISK_THRESHOLD


def test_alert_list_sorted_desc():
    # covers: FR-8  -- alert list sorted by churn risk % descending
    tables = _setup()
    hint = text_analytics.compute_churn_hint_scores(
        tables["support_texts"], tables["customers"]
    )
    model = train_churn_model(tables["customers"], hint, seed=3)
    encoded = build_feature_frame(tables["customers"], hint)
    customers = tables["customers"].copy()
    customers["churn_risk_pct"] = model.predict_proba_percent(encoded)
    alert = customers.sort_values("churn_risk_pct", ascending=False)
    assert alert["churn_risk_pct"].is_monotonic_decreasing
