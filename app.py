"""Streamlit CRM-style dashboard (FR-8, FR-9).

One screen, three blocks:
  1. High-risk alert list sorted by churn risk % descending (FR-8).
  2. SHAP per-customer contribution chart for a selected customer (FR-5).
  3. Top-3 cross-sell recommendations (FR-6, FR-7, FR-22).

Importing this module must NOT start a server; all work is under main(), which
Streamlit invokes. ``streamlit run app.py`` launches the dashboard.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from a clone without installation.
SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import plotly.express as px
import streamlit as st

from churn_recommend import config, data_gen
from churn_recommend.churn_model import build_feature_frame, train_churn_model
from churn_recommend.explain import ChurnExplainer
from churn_recommend.recommend import CrossSellRecommender
from churn_recommend.text_analytics import compute_churn_hint_scores


@st.cache_data(show_spinner=False)
def _load_tables() -> dict:
    """Load committed CSVs, regenerating if absent (offline, NFR-2)."""
    if not config.CUSTOMERS_CSV.exists():
        tables = data_gen.generate_all()
        data_gen.write_csvs(tables)
        return tables
    return data_gen.load_csvs()


@st.cache_resource(show_spinner=False)
def _build_pipeline():
    """Train model + build recommender once and reuse across reruns."""
    tables = _load_tables()
    customers = tables["customers"]
    support_texts = tables["support_texts"]
    purchases = tables["purchases"]

    hint_scores = compute_churn_hint_scores(support_texts, customers)  # FR-3, FR-4
    model = train_churn_model(customers, hint_scores)  # FR-1, FR-21, NFR-3

    encoded = build_feature_frame(customers, hint_scores)
    risk_pct = model.predict_proba_percent(encoded)  # FR-1
    customers = customers.copy()
    customers["churn_risk_pct"] = risk_pct.round(1)

    explainer = ChurnExplainer(model)  # FR-5
    recommender = CrossSellRecommender(purchases)  # FR-6, FR-7, FR-22

    return {
        "customers": customers,
        "encoded": encoded,
        "model": model,
        "explainer": explainer,
        "recommender": recommender,
    }


def main() -> None:
    st.set_page_config(page_title="Churn & Cross-Sell CRM", layout="wide")
    st.title("顧客離脱予兆検知 兼 クロスセル推奨 ダッシュボード")
    st.caption("Churn risk detection + SHAP explanation + cross-sell recommendations")

    pipe = _build_pipeline()
    customers = pipe["customers"]
    model = pipe["model"]

    m = model.metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Holdout AUC", f"{m['auc']:.3f}")  # NFR-3
    c2.metric("Holdout Accuracy", f"{m['accuracy']:.3f}")  # NFR-3
    c3.metric("Churn rate", f"{m['positive_rate'] * 100:.1f}%")

    col_alert, col_shap, col_reco = st.columns([1.1, 1.2, 1.0])

    # ---- Block 1: high-risk alert list, sorted desc (FR-8) ----
    with col_alert:
        st.subheader("1. 高リスク顧客アラート (FR-8)")
        alert = customers.sort_values("churn_risk_pct", ascending=False)
        alert_view = alert[
            ["customer_id", "churn_risk_pct", "contract_type", "support_calls"]
        ].reset_index(drop=True)
        st.dataframe(alert_view, height=420, use_container_width=True)
        selected = st.selectbox("顧客を選択 / select customer", alert_view["customer_id"])

    sel_idx = customers.index[customers["customer_id"] == selected][0]

    # ---- Block 2: SHAP explanation (FR-5) ----
    with col_shap:
        st.subheader("2. SHAP 寄与度 (FR-5)")
        row = pipe["encoded"].loc[[sel_idx]]
        contrib = pipe["explainer"].explain_customer(row, top_n=8)
        fig = px.bar(
            contrib.sort_values("shap_value"),
            x="shap_value",
            y="feature",
            orientation="h",
            color="shap_value",
            color_continuous_scale="RdBu_r",
            title=f"{selected} の離脱要因 (赤=リスク上昇)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---- Block 3: cross-sell recommendations (FR-6, FR-22) ----
    with col_reco:
        st.subheader("3. クロスセル推奨 Top-3 (FR-6)")
        recos = pipe["recommender"].recommend(selected, top_k=config.TOP_K_RECOMMENDATIONS)
        owned = sorted(pipe["recommender"].owned_products(selected))
        st.write("既存契約 (除外済み / FR-22):", ", ".join(owned) if owned else "なし")
        if recos.empty:
            st.info("推奨できる新製品がありません。")
        else:
            st.dataframe(
                recos[["product", "score", "cf_score", "graph_score"]].round(3),
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
