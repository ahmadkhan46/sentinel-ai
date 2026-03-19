from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class AlertOut(BaseModel):
    id: str
    asset_id: str
    org_id: str
    severity: str
    alert_type: str
    title: str
    message: str
    rul_at_alert: float | None
    anomaly_score_at_alert: float | None
    status: str
    acknowledged_by: str | None
    acknowledged_at: datetime | None
    resolved_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


class AlertAcknowledge(BaseModel):
    pass  # body-less; user_id comes from JWT


class AlertResolve(BaseModel):
    resolution_note: str | None = None


class AlertFilter(BaseModel):
    severity: Literal["info", "warning", "critical"] | None = None
    status: Literal["open", "acknowledged", "resolved"] | None = None
    asset_id: str | None = None
    limit: int = 50
    offset: int = 0
