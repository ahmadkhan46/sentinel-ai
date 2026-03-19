from __future__ import annotations

import json
from collections import defaultdict

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = defaultdict(list)

    async def connect(self, ws: WebSocket, org_id: str) -> None:
        await ws.accept()
        self._connections[org_id].append(ws)

    def disconnect(self, ws: WebSocket, org_id: str) -> None:
        self._connections[org_id].discard(ws) if hasattr(
            self._connections[org_id], "discard"
        ) else self._connections[org_id].remove(ws) if ws in self._connections[org_id] else None

    async def broadcast(self, org_id: str, data: dict) -> None:
        dead: list[WebSocket] = []
        for ws in list(self._connections.get(org_id, [])):
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws, org_id)


manager = ConnectionManager()


async def redis_listener(org_id: str) -> None:
    """Subscribe to Redis pub/sub for an org and forward messages to connected WebSocket clients."""
    from api.core.redis import get_redis  # lazy import

    try:
        r = get_redis()
        channel = f"org:{org_id}:inference"
        async with r.pubsub() as pubsub:
            await pubsub.subscribe(channel)
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        await manager.broadcast(org_id, data)
                    except Exception:
                        pass
    except Exception:
        pass  # Redis unavailable — WebSocket live events disabled
