from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class InferenceRequest(BaseModel):
    """Trigger inference for an asset using its latest sensor reading."""
    cycle: int | None = None  # if None, use the latest stored reading


class InferenceOut(BaseModel):
    id: str
    asset_id: str
    model_name: str
    model_version: str
    inferred_at: datetime
    cycle: int | None

    anomaly_score: float | None
    is_anomaly: bool | None
    anomaly_threshold: float | None
    rul_prediction: float | None
    health_index: float | None
    shap_values: dict | None
    feature_importance: dict | None

    model_config = {"from_attributes": True}


class BatchInferenceRequest(BaseModel):
    asset_ids: list[str]
