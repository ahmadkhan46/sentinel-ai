from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SensorReadingCreate(BaseModel):
    cycle: int = Field(..., ge=1)
    op_setting_1: float | None = None
    op_setting_2: float | None = None
    op_setting_3: float | None = None
    sensor_values: dict[str, float] = Field(default_factory=dict)


class SensorReadingOut(BaseModel):
    id: str
    asset_id: str
    cycle: int
    recorded_at: datetime
    op_setting_1: float | None
    op_setting_2: float | None
    op_setting_3: float | None
    sensor_values: dict

    model_config = {"from_attributes": True}


class SensorReadingBatch(BaseModel):
    readings: list[SensorReadingCreate] = Field(..., min_length=1, max_length=1000)
