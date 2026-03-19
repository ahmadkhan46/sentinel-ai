from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.database import get_db
from api.middleware.auth import get_current_user
from api.models.alert import Alert
from api.models.user import User
from api.schemas.alert import AlertOut

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("", response_model=list[AlertOut])
async def list_alerts(
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
    severity: str | None = Query(None),
    alert_status: str | None = Query(None, alias="status"),
    asset_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> list[Alert]:
    q = select(Alert).where(Alert.org_id == current_user.org_id)
    if severity:
        q = q.where(Alert.severity == severity)
    if alert_status:
        q = q.where(Alert.status == alert_status)
    if asset_id:
        q = q.where(Alert.asset_id == asset_id)
    q = q.order_by(Alert.created_at.desc()).offset(offset).limit(limit)
    result = await db.execute(q)
    return list(result.scalars())


@router.post("/{alert_id}/acknowledge", response_model=AlertOut)
async def acknowledge_alert(
    alert_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> Alert:
    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    alert: Alert | None = result.scalar_one_or_none()
    if alert is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")
    if alert.org_id != current_user.org_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    alert.status = "acknowledged"
    alert.acknowledged_by = current_user.id
    alert.acknowledged_at = datetime.now(timezone.utc)
    db.add(alert)
    await db.flush()
    await db.refresh(alert)
    return alert


@router.post("/{alert_id}/resolve", response_model=AlertOut)
async def resolve_alert(
    alert_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> Alert:
    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    alert: Alert | None = result.scalar_one_or_none()
    if alert is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")
    if alert.org_id != current_user.org_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    alert.status = "resolved"
    alert.resolved_at = datetime.now(timezone.utc)
    db.add(alert)
    await db.flush()
    await db.refresh(alert)
    return alert
