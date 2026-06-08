"""LightGBM churn classifier (FR-1, FR-2, FR-4, FR-21, NFR-3).

Trains a binary LightGBM model on customer attributes + the text-derived
``churn_hint_score`` correction feature (FR-4), handling class imbalance via
``is_unbalance`` / ``class_weight`` (FR-21), and reports holdout AUC / Accuracy
(NFR-3). ``predict_proba`` outputs are scaled to 0-100% (FR-1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from . import config


def build_feature_frame(customers: pd.DataFrame, hint_scores: pd.DataFrame) -> pd.DataFrame:
    """Merge the churn-hint text feature into customers and one-hot encode (FR-2, FR-4)."""
    merged = customers.merge(hint_scores, on="customer_id", how="left")
    merged[config.TEXT_FEATURE] = merged[config.TEXT_FEATURE].fillna(0.0)

    encoded = pd.get_dummies(
        merged, columns=config.CATEGORICAL_FEATURES, prefix=config.CATEGORICAL_FEATURES
    )
    return encoded


def _feature_columns(encoded: pd.DataFrame) -> list[str]:
    """All numeric + one-hot categorical + text feature columns."""
    cols = list(config.NUMERIC_FEATURES) + [config.TEXT_FEATURE]
    cat_cols = [
        c
        for c in encoded.columns
        if any(c.startswith(prefix + "_") for prefix in config.CATEGORICAL_FEATURES)
    ]
    return cols + sorted(cat_cols)


@dataclass
class ChurnModel:
    """Wraps a fitted LightGBM churn classifier and its metadata."""

    model: lgb.LGBMClassifier
    feature_columns: list[str]
    metrics: dict = field(default_factory=dict)

    def predict_proba_percent(self, encoded: pd.DataFrame) -> np.ndarray:
        """Return churn probability as 0-100% per row (FR-1)."""
        X = encoded.reindex(columns=self.feature_columns, fill_value=0.0)
        proba = self.model.predict_proba(X)[:, 1]
        return proba * 100.0

    def feature_importance(self) -> pd.Series:
        """LightGBM gain-based feature importance (used to confirm FR-4 contribution)."""
        imp = self.model.booster_.feature_importance(importance_type="gain")
        return pd.Series(imp, index=self.feature_columns).sort_values(ascending=False)


def train_churn_model(
    customers: pd.DataFrame,
    hint_scores: pd.DataFrame,
    test_size: float = 0.25,
    seed: int = config.SEED,
) -> ChurnModel:
    """Train LightGBM with class-imbalance handling and holdout eval (FR-1, FR-21, NFR-3)."""
    encoded = build_feature_frame(customers, hint_scores)
    feat_cols = _feature_columns(encoded)
    X = encoded[feat_cols]
    y = encoded[config.TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # FR-21: class imbalance handling via is_unbalance + balanced class weights.
    clf = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=4,
        min_child_samples=10,
        is_unbalance=True,
        class_weight="balanced",
        random_state=seed,
        verbose=-1,
    )
    clf.fit(X_train, y_train)

    proba_test = clf.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)
    metrics = {
        "auc": float(roc_auc_score(y_test, proba_test)),
        "accuracy": float(accuracy_score(y_test, pred_test)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate": float(y.mean()),
    }

    return ChurnModel(model=clf, feature_columns=feat_cols, metrics=metrics)
