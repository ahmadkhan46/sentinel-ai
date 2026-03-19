from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class WorkOrderCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    description: str
    priority: Literal["low", "medium", "high", "critical"] = "medium"
    alert_id: str | None = None
    assigned_to: str | None = None
    estimated_duration_hours: float | None = None
    scheduled_for: datetime | None = None


class WorkOrderUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    priority: Literal["low", "medium", "high", "critical"] | None = None
    status: Literal["open", "in_progress", "completed", "cancelled"] | None = None
    assigned_to: str | None = None
    estimated_duration_hours: float | None = None
    scheduled_for: datetime | None = None


class WorkOrderOut(BaseModel):
    id: str
    asset_id: str
    org_id: str
    alert_id: str | None
    title: str
    description: str
    priority: str
    status: str
    assigned_to: str | None
    estimated_duration_hours: float | None
    rul_at_creation: int | None
    scheduled_for: datetime | None
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
