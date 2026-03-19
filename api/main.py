from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.core.config import settings
from api.core.database import init_db
from api.core.redis import close_redis


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup
    if settings.ENVIRONMENT == "development":
        await init_db()
    yield
    # Shutdown
    await close_redis()


def create_app() -> FastAPI:
    app = FastAPI(
        title="SENTINEL — Industrial AI Platform",
        description=(
            "Predictive maintenance API powered by NASA C-MAPSS turbofan data. "
            "Provides anomaly detection, RUL prediction, SHAP explanations, and digital twin simulation."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    from api.routers.alerts import router as alert_router
    from api.routers.analytics import router as analytics_router
    from api.routers.assets import router as asset_router
    from api.routers.audit import router as audit_router
    from api.routers.auth import router as auth_router
    from api.routers.inference import router as inference_router
    from api.routers.models import router as model_router
    from api.routers.organisations import router as org_router
    from api.routers.sensors import router as sensor_router
    from api.routers.work_orders import router as wo_router
    from api.routers.ws import router as ws_router

    prefix = "/api/v1"
    app.include_router(auth_router, prefix=prefix)
    app.include_router(org_router, prefix=prefix)
    app.include_router(asset_router, prefix=prefix)
    app.include_router(sensor_router, prefix=prefix)
    app.include_router(inference_router, prefix=prefix)
    app.include_router(alert_router, prefix=prefix)
    app.include_router(analytics_router, prefix=prefix)
    app.include_router(wo_router, prefix=prefix)
    app.include_router(model_router, prefix=prefix)
    app.include_router(audit_router, prefix=prefix)
    app.include_router(ws_router)  # WebSocket has no /api/v1 prefix

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "version": "1.0.0"}

    return app


app = create_app()
