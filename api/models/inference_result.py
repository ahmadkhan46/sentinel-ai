from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.models.base import Base


class InferenceResult(Base):
    __tablename__ = "inference_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    asset_id: Mapped[str] = mapped_column(String(36), ForeignKey("assets.id"), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    inferred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
    cycle: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Anomaly detection
    anomaly_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_anomaly: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    anomaly_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)

    # RUL
    rul_prediction: Mapped[float | None] = mapped_column(Float, nullable=True)
    health_index: Mapped[float | None] = mapped_column(Float, nullable=True)

    # SHAP/LIME explanations stored as JSON
    shap_values: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    feature_importance: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    asset: Mapped[Asset] = relationship("Asset", back_populates="inference_results")


from api.models.asset import Asset  # noqa: E402
