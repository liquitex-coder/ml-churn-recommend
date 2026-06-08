# ML-Churn-Recommend: Churn Prediction + Cross-Sell Recommendation

> Primary README is Japanese: [README.ja.md](README.ja.md)

A runnable demo for marketing / CS / sales budget holders that **detects customer
churn risk early**, **explains WHY** (SHAP), and **auto-generates top-3 cross-sell
recommendations**. It runs straight after `git clone` with **no network access**,
using bundled synthetic data.

## Value proposition

- **Early churn detection** — per-customer churn probability 0–100% (LightGBM).
- **Explainable AI** — SHAP contribution chart showing why risk is high.
- **Automatic cross-sell** — top-3 product recommendations (collaborative
  filtering + graph analysis); already-owned products excluded.
- **CRM-style dashboard** — Streamlit one-screen view: alerts, explanation, recos.

## Architecture

```
                Synthetic data generation (fixed seed / FR-20, NFR-2)
                ┌───────────────────────────────────────────┐
                │ customers.csv  purchases.csv  support_texts │
                └───────────────────────────────────────────┘
                        │                        │
        ┌───────────────┴──────────┐             │
        ▼                          ▼             ▼
  text_analytics              churn_model    recommend
  TF-IDF similarity (FR-3) ─hint─▶ LightGBM   item-item CF
  cancellation-hint score        binary (FR-1) + NetworkX graph
        │                        imbalance(FR-21) (FR-6, FR-22)
        └──────FR-4 correction feature─▶│              │
                                        ▼              │
                                  explain (SHAP)       │
                                  factor viz (FR-5)    │
                                        │              │
                                        ▼              ▼
                       ┌──────────────────────────────────┐
                       │  app.py  Streamlit dashboard       │
                       │  (1) alert list (FR-8)             │
                       │  (2) SHAP contributions (FR-5)     │
                       │  (3) cross-sell recos (FR-6/7/22)  │
                       └──────────────────────────────────┘
```

## Quickstart

```bash
pip install -r requirements.txt
python scripts/generate_data.py   # bundled already; regenerate (reproducible, FR-20)
streamlit run app.py
pytest -q
```

Requires Python 3.10+ (NFR-1).

## Synthetic data (NFR-2, FR-20)

`scripts/generate_data.py` generates three linked tables from a fixed seed,
producing identical output on every run:

- `data/customers.csv` — Telco-like attributes (tenure, monthly charges, login
  frequency, feature usage, payment failures, support calls, contract) + churn
  label. Churners are the minority class (FR-21).
- `data/purchases.csv` — customer × product history with co-purchase bundles.
- `data/support_texts.csv` — 100+ synthetic JP/EN support inquiries tagged with
  sentiment, including cancellation hints ("解約したい", "他社に乗り換え"),
  each linked to a `customer_id`.

Text embedding defaults to TF-IDF (char n-grams, offline). An optional
Sentence-Transformers path is guarded behind a try/except import.

## Requirements traceability

Machine-readable anchors (`FR-x` / `NFR-x`) are defined in:

- [`docs/requirements/A_requirements.md`](docs/requirements/A_requirements.md)
- [`docs/requirements/B_acceptance.md`](docs/requirements/B_acceptance.md)

Each test file carries a mapping header and `# covers: FR-x` markers spanning
FR-1..FR-9, FR-20..FR-22, NFR-1..NFR-3. See the Japanese README for the full
requirement → implementation table.
