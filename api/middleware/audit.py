from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from api.models.audit_log import AuditLog


async def log_action(
    db: AsyncSession,
    org_id: str,
    user_id: str | None,
    action: str,
    resource_type: str | None = None,
    resource_id: str | None = None,
    detail: dict[str, Any] | None = None,
    ip_address: str | None = None,
) -> None:
    entry = AuditLog(
        org_id=org_id,
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        detail=detail,
        ip_address=ip_address,
    )
    db.add(entry)
    # Commit happens in get_db() generator
