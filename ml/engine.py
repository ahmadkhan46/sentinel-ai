"""
ml/engine.py — SentinelEngine

Production inference interface for the SENTINEL platform.
Load a trained ModelBundle and call full_inference() to get typed results
that serialise directly to JSON for the API layer.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from ml.models.rul.xgb_regressor import RULXGB

# ── Result types (Pydantic → direct JSON serialisation) ────────────────────────

class SensorContribution(BaseModel):
    sensor: str
    shap_value: float
    reconstruction_error: float | None = None
    direction: Literal["increasing_risk", "decreasing_risk", "neutral"]


class AnomalyResult(BaseModel):
    anomaly_score: float              # 0-1 normalised via sigmoid on healthy distribution
    raw_score: float                  # raw model output
    is_anomalous: bool
    confidence: float                 # |raw - threshold| / threshold
    triggered_alert: bool
    consecutive_anomalous_cycles: int


class RULResult(BaseModel):
    rul_cycles: float
    rul_hours_estimate: float | None  # None unless cycle_duration_hours is set
    confidence_interval: tuple[float, float]
    nasa_risk_level: Literal["low", "medium", "high", "critical"]


class ExplanationResult(BaseModel):
    top_sensors: list[SensorContribution]
    shap_values: dict[str, float]
    reconstruction_errors: dict[str, float]
    natural_language_summary: str


class InferenceResult(BaseModel):
    anomaly: AnomalyResult
    rul: RULResult
    explanation: ExplanationResult
    asset_id: str | None = None
    model_version: str | None = None
    inference_timestamp: str


# ── ModelBundle ─────────────────────────────────────────────────────────────────

class ModelBundle:
    """All trained artefacts required to run inference for one asset type."""

    def __init__(
        self,
        anomaly_model: Any,
        rul_model: RULXGB,
        scaler: Any,
        feature_cols: list[str],
        threshold: float,
        model_type: str,                      # "iforest" | "ocsvm" | "lstm" | "gru"
        sequence_length: int = 30,
        healthy_score_mean: float = 0.0,
        healthy_score_std: float = 1.0,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        cycle_duration_hours: float | None = None,
        training_mae: float | None = None,
        oc_normaliser: Any | None = None,     # B1: OperatingConditionNormaliser (optional)
    ) -> None:
        self.anomaly_model = anomaly_model
        self.rul_model = rul_model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.threshold = threshold
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.healthy_score_mean = healthy_score_mean
        self.healthy_score_std = healthy_score_std
        self.config = config or {}
        self.metadata = metadata or {}
        self.cycle_duration_hours = cycle_duration_hours
        self.training_mae = training_mae
        self.oc_normaliser = oc_normaliser


# ── Helpers ─────────────────────────────────────────────────────────────────────

_RISK_THRESHOLDS = {"critical": 20, "high": 50, "medium": 100}


def _nasa_risk_level(rul: float) -> Literal["low", "medium", "high", "critical"]:
    if rul < _RISK_THRESHOLDS["critical"]:
        return "critical"
    if rul < _RISK_THRESHOLDS["high"]:
        return "high"
    if rul < _RISK_THRESHOLDS["medium"]:
        return "medium"
    return "low"


# ── SentinelEngine ──────────────────────────────────────────────────────────────

class SentinelEngine:
    """
    Production inference interface for the SENTINEL platform.

    Usage:
        engine = SentinelEngine.from_registry("asset_fd001")
        result = engine.full_inference(sensor_df)   # → InferenceResult
    """

    def __init__(
        self,
        bundle: ModelBundle,
        asset_id: str | None = None,
        model_version: str | None = None,
    ) -> None:
        self._bundle = bundle
        self._asset_id = asset_id
        self._model_version = model_version

    @classmethod
    def from_bundle(
        cls,
        bundle: ModelBundle,
        asset_id: str | None = None,
        model_version: str | None = None,
    ) -> "SentinelEngine":
        """Construct directly from a pre-built ModelBundle."""
        return cls(bundle, asset_id=asset_id, model_version=model_version)

    @classmethod
    def from_registry(
        cls,
        asset_id: str,
        model_version: str = "latest",
        registry_path: str | Path = "models",
    ) -> "SentinelEngine":
        """Load a trained bundle from the ModelRegistry."""
        from ml.models.registry import ModelRegistry  # lazy to avoid circular import
        registry = ModelRegistry(root=Path(registry_path))
        resolved = registry._resolve_version(asset_id, model_version)
        bundle = registry.load(asset_id, version=resolved)
        return cls(bundle, asset_id=asset_id, model_version=resolved)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _preprocess(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Validate columns, apply OC normaliser (if set), then standard scaler."""
        missing = [c for c in self._bundle.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Input DataFrame is missing columns: {missing}")
        out = df.copy()
        # B1: apply operating-condition normalisation first if present
        if self._bundle.oc_normaliser is not None:
            sensor_cols = [c for c in self._bundle.feature_cols if c.startswith("sensor_")]
            out = self._bundle.oc_normaliser.transform(out, sensor_cols)
        out[self._bundle.feature_cols] = self._bundle.scaler.transform(
            out[self._bundle.feature_cols]
        )
        x = out[self._bundle.feature_cols].to_numpy()
        return out, x

    def _normalise_score(self, raw: float) -> float:
        """Map raw anomaly score to [0, 1] via sigmoid on healthy distribution."""
        std = max(self._bundle.healthy_score_std, 1e-9)
        z = (raw - self._bundle.healthy_score_mean) / std
        return float(1.0 / (1.0 + np.exp(-z)))

    def _consecutive_anomalous(self, scores: np.ndarray) -> int:
        """Count trailing consecutive scores at or above the threshold."""
        count = 0
        for s in reversed(scores):
            if s >= self._bundle.threshold:
                count += 1
            else:
                break
        return count

    def _compute_tabular_scores(self, x: np.ndarray) -> np.ndarray:
        return self._bundle.anomaly_model.score(x)

    def _compute_sequence_scores(self, x: np.ndarray) -> np.ndarray:
        """Compute reconstruction errors for all valid windows in x."""
        from ml.models.anomaly.lstm_autoencoder import reconstruction_error
        seq_len = self._bundle.sequence_length
        n = len(x)
        if n < seq_len:
            raise ValueError(
                f"{self._bundle.model_type.upper()} inference requires at least "
                f"{seq_len} cycles; got {n}."
            )
        # Cap to last 100 windows to stay fast at inference time
        max_windows = 100
        start = max(0, n - seq_len - max_windows + 1)
        windows = np.stack([x[i : i + seq_len] for i in range(start, n - seq_len + 1)])
        return reconstruction_error(self._bundle.anomaly_model, windows, device="cpu")

    # ── Public API ───────────────────────────────────────────────────────────

    def score_anomaly(self, sensor_readings: pd.DataFrame) -> AnomalyResult:
        """
        Score the latest sensor readings for anomaly.

        For LSTM/GRU models, sensor_readings must contain at least sequence_length rows.
        Rows are assumed to be sorted oldest-first (ascending cycle order).
        """
        _, x = self._preprocess(sensor_readings)
        model_type = self._bundle.model_type

        if model_type in ("lstm", "gru"):
            all_scores = self._compute_sequence_scores(x)
        else:
            all_scores = self._compute_tabular_scores(x)

        raw = float(all_scores[-1])
        threshold = self._bundle.threshold
        is_anomalous = raw >= threshold

        return AnomalyResult(
            anomaly_score=round(self._normalise_score(raw), 4),
            raw_score=round(raw, 6),
            is_anomalous=is_anomalous,
            confidence=round(abs(raw - threshold) / max(threshold, 1e-9), 4),
            triggered_alert=is_anomalous,
            consecutive_anomalous_cycles=self._consecutive_anomalous(all_scores),
        )

    def predict_rul(self, sensor_readings: pd.DataFrame) -> RULResult:
        """
        Predict Remaining Useful Life from the latest sensor readings.
        Uses the XGBoost regressor on the last row of the (scaled) feature array.
        """
        _, x = self._preprocess(sensor_readings)
        rul_raw = max(0.0, float(self._bundle.rul_model.predict(x[-1:])[0]))

        # Confidence interval: ±1.5 × training MAE (fallback: 15% of prediction)
        mae = self._bundle.training_mae if self._bundle.training_mae is not None else rul_raw * 0.15
        ci_half = mae * 1.5
        ci = (round(max(0.0, rul_raw - ci_half), 1), round(rul_raw + ci_half, 1))

        hours = (
            round(rul_raw * self._bundle.cycle_duration_hours, 1)
            if self._bundle.cycle_duration_hours is not None
            else None
        )

        return RULResult(
            rul_cycles=round(rul_raw, 1),
            rul_hours_estimate=hours,
            confidence_interval=ci,
            nasa_risk_level=_nasa_risk_level(rul_raw),
        )

    def explain(self, sensor_readings: pd.DataFrame) -> ExplanationResult:
        """
        SHAP-based feature attribution + autoencoder reconstruction diagnostics.

        Returns top sensors by impact, raw shap/recon values, and a natural language summary.
        """
        _, x = self._preprocess(sensor_readings)
        feature_cols = self._bundle.feature_cols

        # SHAP for RUL model — uses pred_contribs for XGBoost (no background needed)
        from ml.explain.shap_explain import compute_shap_values, mean_abs_shap

        shap_raw = compute_shap_values(
            self._bundle.rul_model.model,
            x_background=x[:min(200, len(x))],
            x_target=x[-1:],
            silent=True,
        )
        abs_shap = mean_abs_shap(shap_raw)
        shap_dict = {f: round(float(v), 6) for f, v in zip(feature_cols, abs_shap)}

        # Reconstruction errors (LSTM/GRU only)
        recon_dict: dict[str, float] = {}
        if self._bundle.model_type in ("lstm", "gru"):
            seq_len = self._bundle.sequence_length
            if len(x) >= seq_len:
                from ml.explain.recon_error import (
                    sequence_sensor_reconstruction_contrib,
                )
                from ml.models.anomaly.lstm_autoencoder import reconstruct

                window = x[-seq_len:][np.newaxis, :, :]
                recon = reconstruct(self._bundle.anomaly_model, window, device="cpu")
                per_sensor = sequence_sensor_reconstruction_contrib(window, recon)[0]
                recon_dict = {f: round(float(v), 6) for f, v in zip(feature_cols, per_sensor)}

        # Build ranked SensorContribution list
        contributions: list[SensorContribution] = []
        for f in sorted(shap_dict, key=lambda k: abs(shap_dict[k]), reverse=True)[:10]:
            sv = shap_dict[f]
            contributions.append(
                SensorContribution(
                    sensor=f,
                    shap_value=sv,
                    reconstruction_error=recon_dict.get(f) if recon_dict else None,
                    direction=(
                        "increasing_risk" if sv > 0
                        else "decreasing_risk" if sv < 0
                        else "neutral"
                    ),
                )
            )

        # Natural language summary
        if contributions:
            top = contributions[0]
            recon_note = (
                f" (reconstruction error {top.reconstruction_error:.4f})"
                if top.reconstruction_error is not None
                else ""
            )
            top_names = ", ".join(c.sensor for c in contributions[:3])
            summary = (
                f"{top.sensor} is the primary driver of the current prediction{recon_note}. "
                f"Top contributing sensors: {top_names}."
            )
        else:
            summary = "Insufficient data for explanation."

        return ExplanationResult(
            top_sensors=contributions,
            shap_values=shap_dict,
            reconstruction_errors=recon_dict,
            natural_language_summary=summary,
        )

    def full_inference(self, sensor_readings: pd.DataFrame) -> InferenceResult:
        """
        Run anomaly detection + RUL prediction + explainability in one call.
        This is the primary method used by the API inference endpoint.
        """
        return InferenceResult(
            anomaly=self.score_anomaly(sensor_readings),
            rul=self.predict_rul(sensor_readings),
            explanation=self.explain(sensor_readings),
            asset_id=self._asset_id,
            model_version=self._model_version,
            inference_timestamp=datetime.now(timezone.utc).isoformat(),
        )
