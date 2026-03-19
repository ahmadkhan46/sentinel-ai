from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.models.base import Base


class SensorReading(Base):
    __tablename__ = "sensor_readings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    asset_id: Mapped[str] = mapped_column(String(36), ForeignKey("assets.id"), nullable=False, index=True)
    cycle: Mapped[int] = mapped_column(Integer, nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )

    # Operating settings
    op_setting_1: Mapped[float | None] = mapped_column(Float, nullable=True)
    op_setting_2: Mapped[float | None] = mapped_column(Float, nullable=True)
    op_setting_3: Mapped[float | None] = mapped_column(Float, nullable=True)

    # All 21 C-MAPSS sensors stored as JSON for flexibility
    sensor_values: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    asset: Mapped[Asset] = relationship("Asset", back_populates="sensor_readings")


from api.models.asset import Asset  # noqa: E402
