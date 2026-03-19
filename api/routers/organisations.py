from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.database import get_db
from api.middleware.auth import get_current_user, require_role
from api.models.organisation import Organisation
from api.models.user import User
from api.schemas.organisation import OrganisationCreate, OrganisationOut, OrganisationUpdate

router = APIRouter(prefix="/organisations", tags=["organisations"])


@router.post("", response_model=OrganisationOut, status_code=status.HTTP_201_CREATED)
async def create_organisation(
    body: OrganisationCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    _: Annotated[User, Depends(require_role("admin"))],
) -> Organisation:
    existing = await db.execute(select(Organisation).where(Organisation.slug == body.slug))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Slug already in use")

    org = Organisation(**body.model_dump())
    db.add(org)
    await db.flush()
    await db.refresh(org)
    return org


@router.get("/{org_id}", response_model=OrganisationOut)
async def get_organisation(
    org_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> Organisation:
    if current_user.org_id != org_id and current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    result = await db.execute(select(Organisation).where(Organisation.id == org_id))
    org = result.scalar_one_or_none()
    if org is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organisation not found")
    return org


@router.patch("/{org_id}", response_model=OrganisationOut)
async def update_organisation(
    org_id: str,
    body: OrganisationUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(require_role("admin"))],
) -> Organisation:
    result = await db.execute(select(Organisation).where(Organisation.id == org_id))
    org = result.scalar_one_or_none()
    if org is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organisation not found")

    for field, value in body.model_dump(exclude_none=True).items():
        setattr(org, field, value)

    db.add(org)
    await db.flush()
    await db.refresh(org)
    return org
