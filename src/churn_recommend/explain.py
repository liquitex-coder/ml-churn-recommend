"""SHAP per-customer explanations (FR-5).

Uses SHAP's TreeExplainer over the fitted LightGBM model to produce, for any
single customer, the top contributing factors behind their churn risk as a
signed contribution list (positive = pushes risk up).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from .churn_model import ChurnModel


class ChurnExplainer:
    """Wraps a SHAP TreeExplainer for a fitted churn model (FR-5)."""

    def __init__(self, churn_model: ChurnModel):
        self.churn_model = churn_model
        self.explainer = shap.TreeExplainer(churn_model.model)

    def _shap_for_positive_class(self, X: pd.DataFrame) -> np.ndarray:
        """Return a (n_rows, n_features) SHAP matrix for the churn (positive) class."""
        values = self.explainer.shap_values(X)
        if isinstance(values, list):
            # Older SHAP/LightGBM returns [class0, class1].
            arr = np.asarray(values[1])
        else:
            arr = np.asarray(values)
            if arr.ndim == 3:  # (n_rows, n_features, n_classes)
                arr = arr[:, :, -1]
        return arr

    def explain_customer(self, encoded_row: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
        """Top-N signed SHAP contributions for one customer (FR-5).

        ``encoded_row`` is a one-row frame in the same encoded space the model
        was trained on. Returns columns ``[feature, value, shap_value, abs_shap]``
        sorted by absolute contribution descending.
        """
        X = encoded_row.reindex(columns=self.churn_model.feature_columns, fill_value=0.0)
        shap_row = self._shap_for_positive_class(X)[0]

        out = pd.DataFrame(
            {
                "feature": self.churn_model.feature_columns,
                "value": X.iloc[0].to_numpy(),
                "shap_value": shap_row,
            }
        )
        out["abs_shap"] = out["shap_value"].abs()
        out = out.sort_values("abs_shap", ascending=False).reset_index(drop=True)
        return out.head(top_n)
