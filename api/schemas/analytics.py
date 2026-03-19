from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class FleetSummary(BaseModel):
    total_assets: int
    operational: int
    warning: int
    critical: int
    offline: int
    open_alerts: int
    avg_health_index: float | None
    avg_rul: float | None


class AssetTrend(BaseModel):
    asset_id: str
    asset_name: str
    timestamps: list[datetime]
    rul_values: list[float | None]
    health_index_values: list[float | None]
    anomaly_scores: list[float | None]


class MaintenanceMetrics(BaseModel):
    asset_id: str
    mttf: float | None  # mean time to failure (cycles)
    early_warning_cycles: float | None
    oee_impact_pct: float | None
    maintenance_roi: float | None
