from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.database import get_db
from api.middleware.auth import get_current_user
from api.models.alert import Alert
from api.models.asset import Asset
from api.models.inference_result import InferenceResult
from api.models.user import User
from api.schemas.analytics import AssetTrend, FleetSummary

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/fleet", response_model=FleetSummary)
async def fleet_summary(
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> FleetSummary:
    org_id = current_user.org_id

    assets_result = await db.execute(select(Asset).where(Asset.org_id == org_id))
    assets = list(assets_result.scalars())

    counts = {"operational": 0, "warning": 0, "critical": 0, "offline": 0}
    hi_vals = []
    rul_vals = []
    for a in assets:
        counts[a.status] = counts.get(a.status, 0) + 1
        if a.health_index is not None:
            hi_vals.append(a.health_index)
        if a.last_rul is not None:
            rul_vals.append(a.last_rul)

    open_alerts_result = await db.execute(
        select(func.count()).where(Alert.org_id == org_id, Alert.status == "open")
    )
    open_alerts = open_alerts_result.scalar_one()

    return FleetSummary(
        total_assets=len(assets),
        operational=counts["operational"],
        warning=counts["warning"],
        critical=counts["critical"],
        offline=counts["offline"],
        open_alerts=open_alerts,
        avg_health_index=sum(hi_vals) / len(hi_vals) if hi_vals else None,
        avg_rul=sum(rul_vals) / len(rul_vals) if rul_vals else None,
    )


@router.get("/assets/{asset_id}/trend", response_model=AssetTrend)
async def asset_trend(
    asset_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
    limit: int = 200,
) -> AssetTrend:
    result = await db.execute(select(Asset).where(Asset.id == asset_id))
    asset: Asset | None = result.scalar_one_or_none()

    inf_result = await db.execute(
        select(InferenceResult)
        .where(InferenceResult.asset_id == asset_id)
        .order_by(InferenceResult.inferred_at.asc())
        .limit(limit)
    )
    rows = list(inf_result.scalars())

    return AssetTrend(
        asset_id=asset_id,
        asset_name=asset.name if asset else asset_id,
        timestamps=[r.inferred_at for r in rows],
        rul_values=[r.rul_prediction for r in rows],
        health_index_values=[r.health_index for r in rows],
        anomaly_scores=[r.anomaly_score for r in rows],
    )
