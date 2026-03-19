from __future__ import annotations

import numpy as np
import shap


def compute_shap_values(
    model,
    x_background: np.ndarray,
    x_target: np.ndarray,
    silent: bool = True,
):
    # For XGBoost models, use pred_contribs for deterministic and quiet SHAP-like attributions.
    if hasattr(model, "get_booster"):
        import xgboost as xgb

        booster = model.get_booster()
        dmat = xgb.DMatrix(x_target)
        contrib = booster.predict(dmat, pred_contribs=True)
        if contrib.ndim == 2 and contrib.shape[1] == x_target.shape[1] + 1:
            # Last column is bias/base value.
            contrib = contrib[:, :-1]
        return contrib

    explainer = shap.Explainer(model, x_background)
    try:
        return explainer(x_target, silent=silent)
    except TypeError:
        # Backward compatibility for SHAP versions that do not accept `silent`.
        return explainer(x_target)


def mean_abs_shap(values) -> np.ndarray:
    """
    Compute mean absolute SHAP value per feature.
    Supports Explanation objects and raw ndarray values.
    """
    raw = values.values if hasattr(values, "values") else values
    arr = np.asarray(raw)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D SHAP values, got shape={arr.shape}")
    return np.mean(np.abs(arr), axis=0)
