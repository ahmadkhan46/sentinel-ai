from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class OrganisationCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=255)
    slug: str = Field(..., min_length=2, max_length=100, pattern=r"^[a-z0-9-]+$")
    description: str | None = None


class OrganisationUpdate(BaseModel):
    name: str | None = None
    description: str | None = None


class OrganisationOut(BaseModel):
    id: str
    name: str
    slug: str
    description: str | None
    created_at: datetime

    model_config = {"from_attributes": True}
