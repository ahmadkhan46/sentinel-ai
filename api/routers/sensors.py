from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.database import get_db
from api.middleware.auth import get_current_user
from api.models.asset import Asset
from api.models.sensor_reading import SensorReading
from api.models.user import User
from api.schemas.sensor import SensorReadingBatch, SensorReadingCreate, SensorReadingOut

router = APIRouter(prefix="/assets/{asset_id}/sensors", tags=["sensors"])


async def _get_asset_or_404(asset_id: str, db: AsyncSession, user: User) -> Asset:
    result = await db.execute(select(Asset).where(Asset.id == asset_id))
    asset = result.scalar_one_or_none()
    if asset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")
    if asset.org_id != user.org_id and user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    return asset


@router.get("", response_model=list[SensorReadingOut])
async def list_readings(
    asset_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> list[SensorReading]:
    await _get_asset_or_404(asset_id, db, current_user)
    result = await db.execute(
        select(SensorReading)
        .where(SensorReading.asset_id == asset_id)
        .order_by(SensorReading.cycle)
        .offset(offset)
        .limit(limit)
    )
    return list(result.scalars())


@router.post("", response_model=SensorReadingOut, status_code=status.HTTP_201_CREATED)
async def ingest_reading(
    asset_id: str,
    body: SensorReadingCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> SensorReading:
    await _get_asset_or_404(asset_id, db, current_user)
    reading = SensorReading(asset_id=asset_id, **body.model_dump())
    db.add(reading)
    await db.flush()
    await db.refresh(reading)
    return reading


@router.post("/batch", response_model=list[SensorReadingOut], status_code=status.HTTP_201_CREATED)
async def ingest_batch(
    asset_id: str,
    body: SensorReadingBatch,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> list[SensorReading]:
    await _get_asset_or_404(asset_id, db, current_user)
    readings = [SensorReading(asset_id=asset_id, **r.model_dump()) for r in body.readings]
    db.add_all(readings)
    await db.flush()
    for r in readings:
        await db.refresh(r)
    return readings
