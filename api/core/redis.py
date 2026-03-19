from __future__ import annotations

import redis.asyncio as aioredis

from api.core.config import settings

_pool: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    global _pool
    if _pool is None:
        _pool = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    return _pool


async def publish(channel: str, message: str) -> None:
    r = get_redis()
    await r.publish(channel, message)


async def close_redis() -> None:
    global _pool
    if _pool is not None:
        await _pool.aclose()
        _pool = None
