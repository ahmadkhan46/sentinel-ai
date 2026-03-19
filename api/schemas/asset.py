from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class AssetCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    asset_type: str = "turbofan_engine"
    serial_number: str | None = None
    location: str | None = None
    description: str | None = None
    model_name: str | None = None
    model_version: str | None = None


class AssetUpdate(BaseModel):
    name: str | None = None
    location: str | None = None
    description: str | None = None
    status: Literal["operational", "warning", "critical", "offline"] | None = None
    model_name: str | None = None
    model_version: str | None = None


class AssetOut(BaseModel):
    id: str
    org_id: str
    name: str
    asset_type: str
    serial_number: str | None
    location: str | None
    description: str | None
    status: str
    health_index: float | None
    last_rul: int | None
    last_inference_at: datetime | None
    model_name: str | None
    model_version: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
