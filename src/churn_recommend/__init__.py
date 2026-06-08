"""churn_recommend: Churn prediction + cross-sell recommendation demo.

Modules:
    config         - seeds, product catalog, feature lists
    data_gen       - synthetic Telco-like data generation (FR-2, FR-20, NFR-2)
    text_analytics - support-text embedding + churn-hint score (FR-3, FR-4)
    churn_model    - LightGBM churn classifier (FR-1, FR-21, NFR-3)
    explain        - SHAP per-customer explanations (FR-5)
    recommend      - item-item CF + NetworkX cross-sell (FR-6, FR-7, FR-22)
"""

__version__ = "0.1.0"
