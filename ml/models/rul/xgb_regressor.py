from __future__ import annotations

import numpy as np
import pandas as pd


class RULXGB:
    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.03,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError(
                "xgboost is required for RUL baseline. Install dependencies with: pip install -r requirements.txt"
            ) from exc

        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            objective="reg:squarederror",
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)


# ── B2: Monotonic RUL post-processing ─────────────────────────────────────────

def monotonic_rul_predictions(
    raw_preds: np.ndarray,
    val_df: pd.DataFrame,
) -> np.ndarray:
    """
    Enforce the physical constraint that RUL cannot increase over an engine's
    lifetime by applying isotonic regression (non-increasing) per engine.

    Physical motivation: a component cannot "get healthier" during operation;
    RUL can only stay the same or decrease.  Violations in raw XGBoost output
    are corrected by projecting onto the nearest monotonically non-increasing
    sequence via isotonic regression.

    Args:
        raw_preds: XGBoost RUL predictions, shape [n_rows].
        val_df:    DataFrame with 'engine_id' and 'cycle' columns aligned to raw_preds.

    Returns:
        Corrected predictions, same shape as raw_preds.
    """
    from sklearn.isotonic import IsotonicRegression

    out = raw_preds.copy()
    for eid in val_df["engine_id"].unique():
        mask = (val_df["engine_id"] == eid).to_numpy()
        cycles = val_df.loc[val_df["engine_id"] == eid, "cycle"].to_numpy()
        preds = raw_preds[mask]

        sort_idx = np.argsort(cycles)
        corrected = IsotonicRegression(increasing=False).fit_transform(
            cycles[sort_idx], preds[sort_idx]
        )
        # Map corrected values back to the original (unsorted) order
        unsort_idx = np.argsort(sort_idx)
        out[mask] = corrected[unsort_idx]

    return out
