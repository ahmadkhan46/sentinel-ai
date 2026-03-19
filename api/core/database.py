from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from api.core.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.ENVIRONMENT == "development",
    # For SQLite: connect_args prevents "same thread" errors
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create all tables. Run on startup (dev) or use Alembic in production."""
    from api.models.base import Base  # noqa: F401 — import ensures all models are registered
    import api.models.organisation  # noqa: F401
    import api.models.user  # noqa: F401
    import api.models.asset  # noqa: F401
    import api.models.sensor_reading  # noqa: F401
    import api.models.inference_result  # noqa: F401
    import api.models.alert  # noqa: F401
    import api.models.work_order  # noqa: F401
    import api.models.audit_log  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
