from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.database import get_db
from api.middleware.auth import get_current_user, require_role
from api.middleware.audit import log_action
from api.models.asset import Asset
from api.models.user import User
from api.schemas.asset import AssetCreate, AssetOut, AssetUpdate

router = APIRouter(prefix="/assets", tags=["assets"])


def _check_org(current_user: User, org_id: str) -> None:
    if current_user.org_id != org_id and current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")


@router.get("", response_model=list[AssetOut])
async def list_assets(
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> list[Asset]:
    result = await db.execute(
        select(Asset).where(Asset.org_id == current_user.org_id).offset(offset).limit(limit)
    )
    return list(result.scalars())


@router.post("", response_model=AssetOut, status_code=status.HTTP_201_CREATED)
async def create_asset(
    body: AssetCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(require_role("admin", "engineer"))],
) -> Asset:
    asset = Asset(org_id=current_user.org_id, **body.model_dump())
    db.add(asset)
    await db.flush()
    await db.refresh(asset)
    await log_action(db, current_user.org_id, current_user.id, "asset.create", "asset", asset.id)
    return asset


@router.get("/{asset_id}", response_model=AssetOut)
async def get_asset(
    asset_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> Asset:
    result = await db.execute(select(Asset).where(Asset.id == asset_id))
    asset = result.scalar_one_or_none()
    if asset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")
    _check_org(current_user, asset.org_id)
    return asset


@router.patch("/{asset_id}", response_model=AssetOut)
async def update_asset(
    asset_id: str,
    body: AssetUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(require_role("admin", "engineer"))],
) -> Asset:
    result = await db.execute(select(Asset).where(Asset.id == asset_id))
    asset = result.scalar_one_or_none()
    if asset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")
    _check_org(current_user, asset.org_id)

    for field, value in body.model_dump(exclude_none=True).items():
        setattr(asset, field, value)

    db.add(asset)
    await db.flush()
    await db.refresh(asset)
    await log_action(db, current_user.org_id, current_user.id, "asset.update", "asset", asset_id)
    return asset


@router.delete("/{asset_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def delete_asset(
    asset_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(require_role("admin"))],
) -> None:
    result = await db.execute(select(Asset).where(Asset.id == asset_id))
    asset = result.scalar_one_or_none()
    if asset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")
    _check_org(current_user, asset.org_id)
    await db.delete(asset)
    await log_action(db, current_user.org_id, current_user.id, "asset.delete", "asset", asset_id)
