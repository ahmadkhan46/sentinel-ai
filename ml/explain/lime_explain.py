"""
ml/explain/lime_explain.py — LIME explainability alongside SHAP

LIME (Local Interpretable Model-agnostic Explanations) fits a local linear
model around each prediction by perturbing the input.  Running both SHAP and
LIME provides a cross-validation of feature attributions: sensors where both
methods agree are the most reliably identified failure drivers.
"""
from __future__ import annotations

import numpy as np


class LIMEExplainer:
    """
    LIME-based feature attribution for the XGBoost RUL model.

    Usage:
        explainer = LIMEExplainer(feature_cols)
        explainer.fit(x_train)
        lime_attrs = explainer.explain(x_instance, predict_fn)
        report = explainer.compare_with_shap(lime_attrs, shap_dict)
    """

    def __init__(self, feature_cols: list[str], seed: int = 42) -> None:
        self.feature_cols = feature_cols
        self.seed = seed
        self._explainer = None

    def fit(self, x_train: np.ndarray) -> "LIMEExplainer":
        """Fit the LIME explainer on the training feature matrix."""
        from lime import lime_tabular

        self._explainer = lime_tabular.LimeTabularExplainer(
            x_train,
            feature_names=self.feature_cols,
            mode="regression",
            random_state=self.seed,
        )
        return self

    def explain(
        self,
        x_instance: np.ndarray,
        predict_fn,
        num_features: int = 10,
    ) -> dict[str, float]:
        """
        Explain a single prediction.

        Args:
            x_instance: 1-D array of feature values for one observation.
            predict_fn: Callable that takes 2-D array → 1-D predictions.
            num_features: Number of top features to return.

        Returns:
            dict mapping feature name → LIME attribution weight.
        """
        if self._explainer is None:
            raise RuntimeError("Call fit() before explain().")
        exp = self._explainer.explain_instance(
            x_instance,
            predict_fn,
            num_features=num_features,
        )
        return dict(exp.as_list())

    def compare_with_shap(
        self,
        lime_attrs: dict[str, float],
        shap_attrs: dict[str, float],
    ) -> dict[str, dict]:
        """
        Compare LIME and SHAP attributions for the same prediction.

        Returns a dict with per-feature agreement analysis.
        Sensors where both methods agree on direction are the most
        reliable failure drivers — a genuine research finding.
        """
        # LIME feature names may have threshold suffixes like "sensor_11 > 0.34"
        # Strip to bare feature name for alignment
        def _base(name: str) -> str:
            return name.split(" ")[0].split(">")[0].split("<")[0].strip()

        lime_clean = {_base(k): v for k, v in lime_attrs.items()}

        result: dict[str, dict] = {}
        all_features = set(lime_clean) | set(shap_attrs)
        for feat in all_features:
            lv = lime_clean.get(feat, 0.0)
            sv = shap_attrs.get(feat, 0.0)
            lime_sign = int(np.sign(lv))
            shap_sign = int(np.sign(sv))
            result[feat] = {
                "lime_value": round(lv, 6),
                "shap_value": round(sv, 6),
                "agree": lime_sign == shap_sign and lime_sign != 0,
                "lime_direction": "increasing_risk" if lv > 0 else "decreasing_risk" if lv < 0 else "neutral",
                "shap_direction": "increasing_risk" if sv > 0 else "decreasing_risk" if sv < 0 else "neutral",
            }
        return result

    def agreement_summary(self, comparison: dict[str, dict]) -> dict[str, object]:
        """
        Summarise the LIME/SHAP agreement analysis.

        Returns:
            - agreed_features: sensors where both methods agree on direction
            - disagreed_features: sensors where methods disagree
            - agreement_rate: fraction of features with consistent attribution
        """
        agreed = [f for f, v in comparison.items() if v["agree"]]
        disagreed = [f for f, v in comparison.items() if not v["agree"]]
        total = max(len(comparison), 1)
        return {
            "agreed_features": sorted(agreed),
            "disagreed_features": sorted(disagreed),
            "agreement_rate": round(len(agreed) / total, 3),
            "n_features": total,
        }
