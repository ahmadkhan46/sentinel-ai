from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from api.core.database import get_db
from api.core.security import hash_password
from api.main import app
from api.models.base import Base
from api.models.organisation import Organisation
from api.models.user import User

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

engine = create_async_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestSession = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


@pytest_asyncio.fixture(scope="function", autouse=True)
async def setup_db():
    # Import all models so metadata is populated
    import api.models.organisation  # noqa
    import api.models.user  # noqa
    import api.models.asset  # noqa
    import api.models.sensor_reading  # noqa
    import api.models.inference_result  # noqa
    import api.models.alert  # noqa
    import api.models.work_order  # noqa
    import api.models.audit_log  # noqa

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db_session():
    async with TestSession() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def client(db_session: AsyncSession):
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def org_and_admin(db_session: AsyncSession):
    org = Organisation(id="test-org", name="Test Org", slug="test-org")
    db_session.add(org)
    user = User(
        id="test-admin",
        org_id="test-org",
        email="admin@test.com",
        hashed_password=hash_password("Password123!"),
        full_name="Test Admin",
        role="admin",
    )
    db_session.add(user)
    await db_session.commit()
    return org, user
