from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.database import get_db
from api.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_password,
)
from api.middleware.auth import get_current_user
from api.models.user import User
from api.schemas.user import LoginRequest, TokenOut, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenOut)
async def login(body: LoginRequest, db: Annotated[AsyncSession, Depends(get_db)]) -> TokenOut:
    result = await db.execute(select(User).where(User.email == body.email))
    user: User | None = result.scalar_one_or_none()

    if user is None or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account disabled")

    user.last_login = datetime.now(timezone.utc)
    db.add(user)

    return TokenOut(
        access_token=create_access_token(user.id, user.org_id, user.role),
        refresh_token=create_refresh_token(user.id),
    )


@router.post("/refresh", response_model=TokenOut)
async def refresh(refresh_token: str, db: Annotated[AsyncSession, Depends(get_db)]) -> TokenOut:
    try:
        payload = decode_token(refresh_token)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not a refresh token")

    result = await db.execute(select(User).where(User.id == payload["sub"]))
    user: User | None = result.scalar_one_or_none()
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    return TokenOut(
        access_token=create_access_token(user.id, user.org_id, user.role),
        refresh_token=create_refresh_token(user.id),
    )


@router.get("/me", response_model=UserOut)
async def me(current_user: Annotated[User, Depends(get_current_user)]) -> User:
    return current_user
