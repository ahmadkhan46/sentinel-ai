from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.config import settings
from api.core.redis import publish
from api.models.alert import Alert
from api.models.asset import Asset
from api.models.inference_result import InferenceResult
from api.models.sensor_reading import SensorReading


async def run_inference_for_asset(
    asset_id: str,
    db: AsyncSession,
    cycle: int | None = None,
) -> InferenceResult:
    """Load the SentinelEngine for this asset's bound model and score the latest reading."""
    # Fetch asset
    result = await db.execute(select(Asset).where(Asset.id == asset_id))
    asset: Asset | None = result.scalar_one_or_none()
    if asset is None:
        raise ValueError(f"Asset {asset_id} not found")

    model_name = asset.model_name or "fd001"
    model_version = asset.model_version or "latest"

    # Fetch the relevant sensor reading
    query = select(SensorReading).where(SensorReading.asset_id == asset_id)
    if cycle is not None:
        query = query.where(SensorReading.cycle == cycle)
    else:
        query = query.order_by(SensorReading.cycle.desc()).limit(1)

    reading_result = await db.execute(query)
    reading: SensorReading | None = reading_result.scalar_one_or_none()
    if reading is None:
        raise ValueError("No sensor reading available for inference")

    # Build feature vector from sensor_values
    sensor_vals: dict = reading.sensor_values or {}
    feature_array = np.array(list(sensor_vals.values()), dtype=float).reshape(1, -1)

    # Load SentinelEngine (lazy import to avoid startup cost)
    from ml.engine import SentinelEngine  # noqa: PLC0415
    from ml.models.registry import ModelRegistry  # noqa: PLC0415

    registry = ModelRegistry(root=settings.ML_MODEL_PATH)
    engine = SentinelEngine.from_registry(registry, model_name, model_version)

    inference_out = engine.full_inference(feature_array)

    inf_record = InferenceResult(
        asset_id=asset_id,
        model_name=model_name,
        model_version=model_version,
        cycle=reading.cycle,
        anomaly_score=float(inference_out.anomaly_score) if inference_out.anomaly_score is not None else None,
        is_anomaly=bool(inference_out.is_anomaly) if inference_out.is_anomaly is not None else None,
        anomaly_threshold=float(inference_out.threshold) if hasattr(inference_out, "threshold") else None,
        rul_prediction=float(inference_out.rul) if inference_out.rul is not None else None,
        health_index=float(inference_out.health_index) if inference_out.health_index is not None else None,
        shap_values=inference_out.shap_values if inference_out.shap_values else None,
    )
    db.add(inf_record)

    # Update asset state
    asset.health_index = inf_record.health_index
    asset.last_rul = int(inf_record.rul_prediction) if inf_record.rul_prediction is not None else None
    asset.last_inference_at = datetime.now(timezone.utc)
    if inf_record.is_anomaly:
        asset.status = "warning" if (inf_record.rul_prediction or 999) > 30 else "critical"
    db.add(asset)

    await db.flush()
    await db.refresh(inf_record)

    # Raise alert if anomaly detected
    if inf_record.is_anomaly:
        await _create_alert(asset, inf_record, db)

    # Publish real-time event
    await publish(
        f"org:{asset.org_id}:inference",
        json.dumps({"asset_id": asset_id, "rul": inf_record.rul_prediction, "is_anomaly": inf_record.is_anomaly}),
    )

    return inf_record


async def _create_alert(asset: Asset, inf: InferenceResult, db: AsyncSession) -> None:
    rul = inf.rul_prediction
    severity = "critical" if rul is not None and rul < 30 else "warning"
    alert = Alert(
        asset_id=asset.id,
        org_id=asset.org_id,
        severity=severity,
        alert_type="anomaly",
        title=f"Anomaly detected on {asset.name}",
        message=(
            f"Anomaly score {inf.anomaly_score:.3f} exceeded threshold. "
            f"Estimated RUL: {rul:.0f} cycles." if rul else "RUL unavailable."
        ),
        rul_at_alert=rul,
        anomaly_score_at_alert=inf.anomaly_score,
    )
    db.add(alert)
