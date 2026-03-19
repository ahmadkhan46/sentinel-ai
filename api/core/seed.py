"""Demo data seeder — run once to populate a fresh SQLite database."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from api.core.database import AsyncSessionLocal, init_db
from api.core.security import hash_password
from api.models.alert import Alert
from api.models.asset import Asset
from api.models.organisation import Organisation
from api.models.sensor_reading import SensorReading
from api.models.user import User
from api.models.work_order import WorkOrder

ORG_ID = "org-acme-001"
USER_ADMIN_ID = "user-admin-001"
USER_ENG_ID = "user-eng-001"
USER_VIEW_ID = "user-view-001"

ASSET_IDS = [
    "asset-engine-fd001",
    "asset-engine-fd002",
    "asset-engine-fd003",
    "asset-engine-fd004",
]

# C-MAPSS sensor names
SENSOR_NAMES = [f"sensor_{i}" for i in range(1, 22)]


async def seed(db: AsyncSession) -> None:
    # Organisation
    org = Organisation(
        id=ORG_ID,
        name="Acme Manufacturing",
        slug="acme-manufacturing",
        description="Demo industrial fleet — C-MAPSS turbofan engines",
    )
    db.add(org)

    # Users
    users = [
        User(
            id=USER_ADMIN_ID,
            org_id=ORG_ID,
            email="admin@acme.com",
            hashed_password=hash_password("Admin1234!"),
            full_name="Alex Admin",
            role="admin",
        ),
        User(
            id=USER_ENG_ID,
            org_id=ORG_ID,
            email="engineer@acme.com",
            hashed_password=hash_password("Engineer1234!"),
            full_name="Evan Engineer",
            role="engineer",
        ),
        User(
            id=USER_VIEW_ID,
            org_id=ORG_ID,
            email="viewer@acme.com",
            hashed_password=hash_password("Viewer1234!"),
            full_name="Val Viewer",
            role="viewer",
        ),
    ]
    db.add_all(users)

    # Assets (one per C-MAPSS subset)
    assets = [
        Asset(
            id=ASSET_IDS[0],
            org_id=ORG_ID,
            name="Turbofan Engine FD001",
            asset_type="turbofan_engine",
            serial_number="TF-FD001",
            location="Plant A — Bay 1",
            status="operational",
            health_index=0.82,
            last_rul=145,
            model_name="fd001",
            model_version="latest",
        ),
        Asset(
            id=ASSET_IDS[1],
            org_id=ORG_ID,
            name="Turbofan Engine FD002",
            asset_type="turbofan_engine",
            serial_number="TF-FD002",
            location="Plant A — Bay 2",
            status="warning",
            health_index=0.54,
            last_rul=42,
            model_name="fd002",
            model_version="latest",
        ),
        Asset(
            id=ASSET_IDS[2],
            org_id=ORG_ID,
            name="Turbofan Engine FD003",
            asset_type="turbofan_engine",
            serial_number="TF-FD003",
            location="Plant B — Bay 1",
            status="operational",
            health_index=0.71,
            last_rul=88,
            model_name="fd003",
            model_version="latest",
        ),
        Asset(
            id=ASSET_IDS[3],
            org_id=ORG_ID,
            name="Turbofan Engine FD004",
            asset_type="turbofan_engine",
            serial_number="TF-FD004",
            location="Plant B — Bay 2",
            status="critical",
            health_index=0.23,
            last_rul=12,
            model_name="fd004",
            model_version="latest",
        ),
    ]
    db.add_all(assets)

    # 10 sensor readings per asset (cycles 1–10)
    readings = []
    import random
    rng = random.Random(42)
    for asset in assets:
        for cycle in range(1, 11):
            readings.append(
                SensorReading(
                    asset_id=asset.id,
                    cycle=cycle,
                    op_setting_1=round(rng.uniform(-0.01, 0.01), 4),
                    op_setting_2=round(rng.uniform(-0.0003, 0.0003), 6),
                    op_setting_3=round(rng.choice([60.0, 80.0, 100.0]), 1),
                    sensor_values={s: round(rng.uniform(100, 600), 2) for s in SENSOR_NAMES},
                )
            )
    db.add_all(readings)

    # 3 alerts
    alerts = [
        Alert(
            asset_id=ASSET_IDS[1],
            org_id=ORG_ID,
            severity="warning",
            alert_type="rul_threshold",
            title="RUL below 50 cycles — FD002",
            message="Engine FD002 estimated RUL dropped to 42 cycles. Schedule maintenance.",
            rul_at_alert=42.0,
            anomaly_score_at_alert=0.71,
            status="open",
        ),
        Alert(
            asset_id=ASSET_IDS[3],
            org_id=ORG_ID,
            severity="critical",
            alert_type="anomaly",
            title="Critical anomaly detected — FD004",
            message="Anomaly score 0.94 exceeded threshold. RUL: 12 cycles. Immediate inspection required.",
            rul_at_alert=12.0,
            anomaly_score_at_alert=0.94,
            status="acknowledged",
            acknowledged_by=USER_ENG_ID,
            acknowledged_at=datetime.now(timezone.utc),
        ),
        Alert(
            asset_id=ASSET_IDS[0],
            org_id=ORG_ID,
            severity="info",
            alert_type="health_degradation",
            title="Health index declining — FD001",
            message="Engine FD001 health index fell from 0.91 to 0.82 over last 20 cycles.",
            rul_at_alert=145.0,
            anomaly_score_at_alert=0.31,
            status="resolved",
            resolved_at=datetime.now(timezone.utc),
        ),
    ]
    db.add_all(alerts)

    # 1 work order
    wo = WorkOrder(
        asset_id=ASSET_IDS[3],
        org_id=ORG_ID,
        title="Emergency inspection — FD004",
        description=(
            "Critical anomaly detected. Inspect high-pressure compressor section. "
            "Check blade wear and bearing condition. RUL at creation: 12 cycles."
        ),
        priority="critical",
        status="in_progress",
        assigned_to=USER_ENG_ID,
        estimated_duration_hours=8.0,
        rul_at_creation=12,
    )
    db.add(wo)

    await db.commit()
    print("Seed complete.")
    print(f"  Admin:    admin@acme.com / Admin1234!")
    print(f"  Engineer: engineer@acme.com / Engineer1234!")
    print(f"  Viewer:   viewer@acme.com / Viewer1234!")


async def main() -> None:
    await init_db()
    async with AsyncSessionLocal() as db:
        await seed(db)


if __name__ == "__main__":
    asyncio.run(main())
