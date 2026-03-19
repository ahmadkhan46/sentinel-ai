from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.database import get_db
from api.middleware.audit import log_action
from api.middleware.auth import get_current_user, require_role
from api.models.asset import Asset
from api.models.user import User
from api.models.work_order import WorkOrder
from api.schemas.work_order import WorkOrderCreate, WorkOrderOut, WorkOrderUpdate

router = APIRouter(prefix="/assets/{asset_id}/work-orders", tags=["work_orders"])


async def _check_asset(asset_id: str, db: AsyncSession, user: User) -> Asset:
    result = await db.execute(select(Asset).where(Asset.id == asset_id))
    asset = result.scalar_one_or_none()
    if asset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")
    if asset.org_id != user.org_id and user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    return asset


@router.get("", response_model=list[WorkOrderOut])
async def list_work_orders(
    asset_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> list[WorkOrder]:
    await _check_asset(asset_id, db, current_user)
    result = await db.execute(
        select(WorkOrder)
        .where(WorkOrder.asset_id == asset_id)
        .order_by(WorkOrder.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    return list(result.scalars())


@router.post("", response_model=WorkOrderOut, status_code=status.HTTP_201_CREATED)
async def create_work_order(
    asset_id: str,
    body: WorkOrderCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(require_role("admin", "engineer"))],
) -> WorkOrder:
    asset = await _check_asset(asset_id, db, current_user)
    wo = WorkOrder(
        asset_id=asset_id,
        org_id=asset.org_id,
        rul_at_creation=asset.last_rul,
        **body.model_dump(),
    )
    db.add(wo)
    await db.flush()
    await db.refresh(wo)
    await log_action(db, asset.org_id, current_user.id, "work_order.create", "work_order", wo.id)
    return wo


@router.patch("/{wo_id}", response_model=WorkOrderOut)
async def update_work_order(
    asset_id: str,
    wo_id: str,
    body: WorkOrderUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(require_role("admin", "engineer"))],
) -> WorkOrder:
    await _check_asset(asset_id, db, current_user)
    result = await db.execute(select(WorkOrder).where(WorkOrder.id == wo_id, WorkOrder.asset_id == asset_id))
    wo = result.scalar_one_or_none()
    if wo is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Work order not found")

    updates = body.model_dump(exclude_none=True)
    if updates.get("status") == "completed" and wo.completed_at is None:
        updates["completed_at"] = datetime.now(timezone.utc)

    for field, value in updates.items():
        setattr(wo, field, value)

    db.add(wo)
    await db.flush()
    await db.refresh(wo)
    return wo
