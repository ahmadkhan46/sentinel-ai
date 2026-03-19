from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.database import get_db
from api.middleware.auth import get_current_user
from api.models.asset import Asset
from api.models.inference_result import InferenceResult
from api.models.user import User
from api.schemas.inference import InferenceOut, InferenceRequest
from api.services.inference_service import run_inference_for_asset

router = APIRouter(prefix="/assets/{asset_id}/inference", tags=["inference"])


async def _check_asset(asset_id: str, db: AsyncSession, user: User) -> Asset:
    result = await db.execute(select(Asset).where(Asset.id == asset_id))
    asset = result.scalar_one_or_none()
    if asset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")
    if asset.org_id != user.org_id and user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    return asset


@router.post("", response_model=InferenceOut, status_code=status.HTTP_201_CREATED)
async def trigger_inference(
    asset_id: str,
    body: InferenceRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> InferenceResult:
    await _check_asset(asset_id, db, current_user)
    try:
        return await run_inference_for_asset(asset_id, db, cycle=body.cycle)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))


@router.get("", response_model=list[InferenceOut])
async def list_results(
    asset_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> list[InferenceResult]:
    await _check_asset(asset_id, db, current_user)
    result = await db.execute(
        select(InferenceResult)
        .where(InferenceResult.asset_id == asset_id)
        .order_by(InferenceResult.inferred_at.desc())
        .offset(offset)
        .limit(limit)
    )
    return list(result.scalars())
