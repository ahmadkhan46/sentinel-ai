from __future__ import annotations

import asyncio

from api.workers.celery_app import celery


@celery.task(name="inference.run", bind=True, max_retries=3)
def run_inference_task(self, asset_id: str, cycle: int | None = None) -> dict:
    """Celery task wrapper around the async inference service."""
    from api.core.database import AsyncSessionLocal
    from api.services.inference_service import run_inference_for_asset

    async def _run() -> dict:
        async with AsyncSessionLocal() as db:
            try:
                result = await run_inference_for_asset(asset_id, db, cycle=cycle)
                await db.commit()
                return {
                    "inference_id": result.id,
                    "anomaly_score": result.anomaly_score,
                    "is_anomaly": result.is_anomaly,
                    "rul": result.rul_prediction,
                }
            except Exception:
                await db.rollback()
                raise

    try:
        return asyncio.run(_run())
    except Exception as exc:
        raise self.retry(exc=exc, countdown=5)
