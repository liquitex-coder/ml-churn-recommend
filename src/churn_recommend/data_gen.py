"""Synthetic data generation (FR-2, FR-20, NFR-2).

Produces three linked tables, fully offline and reproducible from a fixed
seed (FR-20):

* customers      - Telco-like attributes, usage, payments, support counts and
                   a churn label (FR-2). Class imbalance baked in (minority
                   churners) so FR-21 handling is meaningful.
* purchases      - customer x product purchase history for the recommender.
* support_texts  - >= 100 synthetic JP/EN support inquiries tagged with a
                   sentiment, including explicit cancellation-hint phrases
                   (NFR-2), each linked to a customer_id.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# ---------------------------------------------------------------------------
# Synthetic support-text templates (JP primary + a few EN).
# Each tuple is (sentiment, text). "cancellation" texts are the churn hints.
# ---------------------------------------------------------------------------
CANCELLATION_TEXTS = [
    "解約したいのですが手続きを教えてください。",
    "解約方法を教えてください。",
    "他社に乗り換えを検討しています。",
    "料金が高いので解約を考えています。",
    "サービスに不満があり継続するか迷っています。",
    "もうこのサービスは使わないので止めたいです。",
    "competitor の方が安いので乗り換えたい。",
    "I want to cancel my subscription, it is too expensive.",
    "解約の違約金はいくらですか？",
    "ずっと繋がらない、これでは解約せざるを得ない。",
]

NEGATIVE_TEXTS = [
    "ログインできず非常に困っています。",
    "今日も障害が起きていて不満です。",
    "サポートの対応が遅くて困ります。",
    "請求金額が間違っている気がします。",
    "The app keeps crashing and it is frustrating.",
]

NEUTRAL_TEXTS = [
    "パスワードの再設定方法を教えてください。",
    "請求書の発行日はいつですか？",
    "プランの変更方法について質問です。",
    "領収書がほしいのですが発行できますか？",
    "How do I update my payment method?",
]

POSITIVE_TEXTS = [
    "とても満足しています、ありがとうございます。",
    "サポートが丁寧で助かりました。",
    "新機能がとても便利です、満足しています。",
    "Great service, very happy with the support!",
    "おすすめプランを教えてもらえて満足です。",
]

SENTIMENT_POOLS = {
    "cancellation": CANCELLATION_TEXTS,
    "negative": NEGATIVE_TEXTS,
    "neutral": NEUTRAL_TEXTS,
    "positive": POSITIVE_TEXTS,
}


def generate_customers(n_customers: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate Telco-like customer attributes + churn label (FR-2, FR-21)."""
    customer_id = [f"C{idx:05d}" for idx in range(n_customers)]

    tenure = rng.integers(1, 72, size=n_customers)
    monthly_charges = np.round(rng.uniform(20, 120, size=n_customers), 2)
    total_charges = np.round(monthly_charges * tenure * rng.uniform(0.8, 1.1, size=n_customers), 2)
    login_freq = np.round(rng.gamma(2.0, 2.0, size=n_customers), 2)
    feature_usage = np.round(rng.uniform(0, 100, size=n_customers), 2)
    payment_failures = rng.poisson(0.4, size=n_customers)
    support_calls = rng.poisson(1.2, size=n_customers)
    contract = rng.choice(config.CONTRACT_TYPES, size=n_customers, p=[0.55, 0.25, 0.20])
    payment_method = rng.choice(config.PAYMENT_METHODS, size=n_customers)

    # Churn risk as a logistic function of risk drivers -> minority churners.
    contract_risk = np.where(contract == "Month-to-month", 1.0, 0.0)
    z = (
        -2.6
        + 0.045 * (monthly_charges - 70)
        - 0.03 * (tenure - 30)
        - 0.18 * (login_freq - 4)
        - 0.015 * (feature_usage - 50)
        + 0.55 * payment_failures
        + 0.35 * support_calls
        + 1.1 * contract_risk
    )
    prob = 1.0 / (1.0 + np.exp(-z))
    churned = (rng.uniform(size=n_customers) < prob).astype(int)

    return pd.DataFrame(
        {
            "customer_id": customer_id,
            "tenure_months": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "login_freq_per_week": login_freq,
            "feature_usage_score": feature_usage,
            "payment_failures": payment_failures,
            "support_calls": support_calls,
            "contract_type": contract,
            "payment_method": payment_method,
            "churned": churned,
        }
    )


def generate_purchases(customers: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Generate customer x product purchase history (FR-6 input).

    Higher-tenure / higher-spend customers tend to own more products, with a
    couple of correlated product bundles so item-item CF has signal.
    """
    products = config.PRODUCTS
    bundles = [
        ("Fiber_Internet", "Online_Security", "Online_Backup"),
        ("Streaming_TV", "Streaming_Music"),
        ("Tech_Support", "Premium_Support", "Device_Protection"),
        ("Phone_Line", "Cloud_Storage"),
    ]

    rows = []
    for _, cust in customers.iterrows():
        base = 1 + int(cust["tenure_months"] / 18) + int(cust["monthly_charges"] / 50)
        n_buy = min(len(products), max(1, base))
        owned: set[str] = set()

        # Seed with a bundle to create co-purchase structure.
        bundle = bundles[rng.integers(0, len(bundles))]
        for p in bundle:
            if len(owned) < n_buy:
                owned.add(p)
        # Fill the rest randomly.
        while len(owned) < n_buy:
            owned.add(products[rng.integers(0, len(products))])

        for p in owned:
            rows.append({"customer_id": cust["customer_id"], "product": p, "quantity": 1})

    return pd.DataFrame(rows)


def generate_support_texts(
    customers: pd.DataFrame, n_texts: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Generate >= 100 synthetic support inquiries linked to customers (NFR-2, FR-3).

    Churned customers are more likely to have produced cancellation-hint texts.
    """
    customer_ids = customers["customer_id"].to_numpy()
    churn_map = dict(zip(customers["customer_id"], customers["churned"]))

    rows = []
    for idx in range(n_texts):
        cid = customer_ids[rng.integers(0, len(customer_ids))]
        if churn_map[cid] == 1:
            sentiment = rng.choice(
                ["cancellation", "negative", "neutral", "positive"],
                p=[0.45, 0.30, 0.20, 0.05],
            )
        else:
            sentiment = rng.choice(
                ["cancellation", "negative", "neutral", "positive"],
                p=[0.05, 0.15, 0.45, 0.35],
            )
        pool = SENTIMENT_POOLS[sentiment]
        text = pool[rng.integers(0, len(pool))]
        rows.append(
            {
                "text_id": f"T{idx:05d}",
                "customer_id": cid,
                "text": text,
                "sentiment": sentiment,
            }
        )

    return pd.DataFrame(rows)


def generate_all(
    n_customers: int = config.N_CUSTOMERS,
    n_texts: int = config.N_SUPPORT_TEXTS,
    seed: int = config.SEED,
) -> dict[str, pd.DataFrame]:
    """Generate all three tables reproducibly from a fixed seed (FR-20)."""
    rng = np.random.default_rng(seed)
    customers = generate_customers(n_customers, rng)
    purchases = generate_purchases(customers, rng)
    support_texts = generate_support_texts(customers, n_texts, rng)
    return {
        "customers": customers,
        "purchases": purchases,
        "support_texts": support_texts,
    }


def write_csvs(tables: dict[str, pd.DataFrame], data_dir: Path = config.DATA_DIR) -> None:
    """Persist the generated tables to data/*.csv."""
    data_dir.mkdir(parents=True, exist_ok=True)
    tables["customers"].to_csv(config.CUSTOMERS_CSV, index=False)
    tables["purchases"].to_csv(config.PURCHASES_CSV, index=False)
    tables["support_texts"].to_csv(config.SUPPORT_TEXTS_CSV, index=False)


def load_csvs(data_dir: Path = config.DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load the committed sample CSVs (offline, NFR-2)."""
    return {
        "customers": pd.read_csv(config.CUSTOMERS_CSV),
        "purchases": pd.read_csv(config.PURCHASES_CSV),
        "support_texts": pd.read_csv(config.SUPPORT_TEXTS_CSV),
    }
