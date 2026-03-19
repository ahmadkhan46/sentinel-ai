from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, status
from pydantic import BaseModel

from api.core.config import settings
from api.middleware.auth import require_role
from api.models.user import User

router = APIRouter(prefix="/models", tags=["models"])


class ModelVersionOut(BaseModel):
    name: str
    versions: list[str]


class TrainRequest(BaseModel):
    config: str = "configs/fd001.yaml"


@router.get("", response_model=list[ModelVersionOut])
async def list_models(
    _: Annotated[User, Depends(require_role("admin", "engineer"))],
) -> list[ModelVersionOut]:
    from ml.models.registry import ModelRegistry

    registry = ModelRegistry(root=settings.ML_MODEL_PATH)
    # list_versions returns {model_name: [v1, v2, ...]}
    try:
        versions = registry.list_versions()
    except Exception:
        versions = {}

    return [ModelVersionOut(name=name, versions=vs) for name, vs in versions.items()]


@router.post("/train", status_code=status.HTTP_202_ACCEPTED)
async def trigger_training(
    body: TrainRequest,
    background_tasks: BackgroundTasks,
    _: Annotated[User, Depends(require_role("admin"))],
) -> dict:
    from api.workers.training_tasks import run_training_pipeline

    task = run_training_pipeline.delay(config=body.config)
    return {"task_id": task.id, "status": "queued"}
