from __future__ import annotations

import pytest
from httpx import AsyncClient


async def _get_token(client: AsyncClient) -> str:
    r = await client.post(
        "/api/v1/auth/login",
        json={"email": "admin@test.com", "password": "Password123!"},
    )
    return r.json()["access_token"]


@pytest.mark.asyncio
async def test_list_assets_empty(client: AsyncClient, org_and_admin):
    token = await _get_token(client)
    r = await client.get("/api/v1/assets", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json() == []


@pytest.mark.asyncio
async def test_create_and_get_asset(client: AsyncClient, org_and_admin):
    token = await _get_token(client)
    headers = {"Authorization": f"Bearer {token}"}

    create_r = await client.post(
        "/api/v1/assets",
        json={"name": "Test Engine", "asset_type": "turbofan_engine", "location": "Bay 1"},
        headers=headers,
    )
    assert create_r.status_code == 201
    asset_id = create_r.json()["id"]

    get_r = await client.get(f"/api/v1/assets/{asset_id}", headers=headers)
    assert get_r.status_code == 200
    assert get_r.json()["name"] == "Test Engine"


@pytest.mark.asyncio
async def test_update_asset(client: AsyncClient, org_and_admin):
    token = await _get_token(client)
    headers = {"Authorization": f"Bearer {token}"}

    create_r = await client.post(
        "/api/v1/assets",
        json={"name": "Engine X"},
        headers=headers,
    )
    asset_id = create_r.json()["id"]

    patch_r = await client.patch(
        f"/api/v1/assets/{asset_id}",
        json={"status": "warning"},
        headers=headers,
    )
    assert patch_r.status_code == 200
    assert patch_r.json()["status"] == "warning"
