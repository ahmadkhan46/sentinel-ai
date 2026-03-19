from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.models.base import Base


class Asset(Base):
    __tablename__ = "assets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    org_id: Mapped[str] = mapped_column(String(36), ForeignKey("organisations.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    asset_type: Mapped[str] = mapped_column(String(100), nullable=False, default="turbofan_engine")
    serial_number: Mapped[str | None] = mapped_column(String(100), unique=True, nullable=True)
    location: Mapped[str | None] = mapped_column(String(255), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Operational state
    status: Mapped[str] = mapped_column(String(50), default="operational")  # operational|warning|critical|offline
    health_index: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_rul: Mapped[int | None] = mapped_column(Integer, nullable=True)  # cycles
    last_inference_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # ML model binding
    model_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(50), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    organisation: Mapped[Organisation] = relationship("Organisation", back_populates="assets")
    sensor_readings: Mapped[list[SensorReading]] = relationship("SensorReading", back_populates="asset")
    inference_results: Mapped[list[InferenceResult]] = relationship("InferenceResult", back_populates="asset")
    alerts: Mapped[list[Alert]] = relationship("Alert", back_populates="asset")
    work_orders: Mapped[list[WorkOrder]] = relationship("WorkOrder", back_populates="asset")


from api.models.alert import Alert  # noqa: E402
from api.models.inference_result import InferenceResult  # noqa: E402
from api.models.organisation import Organisation  # noqa: E402
from api.models.sensor_reading import SensorReading  # noqa: E402
from api.models.work_order import WorkOrder  # noqa: E402
