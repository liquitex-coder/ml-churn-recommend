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

import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from churn_recommend import config, data_gen  # noqa: E402
from churn_recommend.churn_model import build_feature_frame, train_churn_model  # noqa: E402
from churn_recommend.explain import ChurnExplainer  # noqa: E402
from churn_recommend.recommend import CrossSellRecommender  # noqa: E402
from churn_recommend.text_analytics import compute_churn_hint_scores  # noqa: E402


@st.cache_data(show_spinner=False)
def _load_tables() -> dict:
    """Load committed CSVs, regenerating if absent (offline, NFR-2)."""
    if not config.CUSTOMERS_CSV.exists():
        tables = data_gen.generate_all()
        data_gen.write_csvs(tables)
        return tables
    return data_gen.load_csvs()


def _pipeline_core(tables: dict) -> dict:
    """Train model + build recommender from a tables dict (sample or uploaded)."""
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


@st.cache_resource(show_spinner=False)
def _build_pipeline():
    """Sample path: train once and reuse across reruns."""
    return _pipeline_core(_load_tables())


def _pipeline_from_upload() -> dict:
    """Upload path: read user CSVs (customers required; purchases/support optional,
    falling back to the sample), validate, and build the pipeline fresh."""
    sample = _load_tables()
    st.sidebar.download_button(
        "顧客テンプレCSV / customers template",
        sample["customers"].to_csv(index=False).encode("utf-8"),
        file_name="customers_template.csv", mime="text/csv",
        help="この列構成に合わせてアップロードしてください。",
    )
    st.sidebar.caption("顧客の必須列: " + ", ".join(data_gen.REQUIRED_CUSTOMER_COLUMNS))
    cust_f = st.sidebar.file_uploader("顧客CSV / customers（必須）", type="csv", key="cust")
    pur_f = st.sidebar.file_uploader("購買CSV / purchases（任意）", type="csv", key="pur")
    sup_f = st.sidebar.file_uploader("問い合わせCSV / support_texts（任意）", type="csv", key="sup")

    if cust_f is None:
        st.info("顧客CSVをアップロードするか、左で『サンプルデータ』を選んでください。"
                " / Upload a customers CSV or pick 'Sample'.")
        st.stop()
    try:
        customers = data_gen.read_customers_csv(cust_f)
        purchases = data_gen.read_purchases_csv(pur_f) if pur_f else sample["purchases"]
        support = data_gen.read_support_csv(sup_f) if sup_f else sample["support_texts"]
    except ValueError as e:
        st.error(str(e))
        st.stop()
    st.sidebar.success(f"顧客 / customers: {len(customers):,} 件読込")
    return _pipeline_core(
        {"customers": customers, "purchases": purchases, "support_texts": support}
    )


def main() -> None:
    st.set_page_config(page_title="Churn & Cross-Sell CRM", layout="wide")
    st.title("顧客離脱予兆検知 兼 クロスセル推奨 ダッシュボード")
    st.caption("Churn risk detection + SHAP explanation + cross-sell recommendations")

    # ---- Data source: sample (default) or user upload ----
    st.sidebar.header("データソース / Data source")
    source = st.sidebar.radio(
        "入力データ / Input data",
        ["サンプルデータ / Sample", "CSVアップロード / Upload CSV"],
        help="自社の顧客データをアップロードするか、同梱サンプルで試せます。",
    )
    if source.startswith("CSV"):
        pipe = _pipeline_from_upload()
    else:
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
