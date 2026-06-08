"""Central configuration: seeds, sizes, product catalog, feature lists.

Keeping these in one place makes the synthetic data reproducible (FR-20)
and gives the model / recommender a single source of truth.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility (FR-20)
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Dataset sizes (kept small so tests run in seconds; NFR-2)
# ---------------------------------------------------------------------------
N_CUSTOMERS = 400
N_SUPPORT_TEXTS = 140  # >= 100 synthetic support inquiries (NFR-2)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CUSTOMERS_CSV = DATA_DIR / "customers.csv"
PURCHASES_CSV = DATA_DIR / "purchases.csv"
SUPPORT_TEXTS_CSV = DATA_DIR / "support_texts.csv"

# ---------------------------------------------------------------------------
# Product catalog for cross-sell (FR-6) — 10 products/options
# ---------------------------------------------------------------------------
PRODUCTS = [
    "Fiber_Internet",
    "Phone_Line",
    "Streaming_TV",
    "Streaming_Music",
    "Online_Security",
    "Online_Backup",
    "Device_Protection",
    "Tech_Support",
    "Cloud_Storage",
    "Premium_Support",
]

CONTRACT_TYPES = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHODS = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]

# ---------------------------------------------------------------------------
# Feature lists used by the churn model (FR-2)
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "login_freq_per_week",
    "feature_usage_score",
    "payment_failures",
    "support_calls",
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "payment_method",
]

# Text-derived correction feature folded into the churn model (FR-4)
TEXT_FEATURE = "churn_hint_score"

# Full feature list consumed by churn_model after encoding (FR-2, FR-4)
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE]

TARGET = "churned"

# Risk threshold (%) above which a customer is flagged on the alert list (FR-8)
HIGH_RISK_THRESHOLD = 50.0

TOP_K_RECOMMENDATIONS = 3  # FR-6
