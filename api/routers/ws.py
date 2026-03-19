from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.services.websocket_service import manager, redis_listener

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/{org_id}")
async def websocket_endpoint(websocket: WebSocket, org_id: str) -> None:
    await manager.connect(websocket, org_id)
    # Start a Redis listener task for this org if not already running
    listener_task = asyncio.create_task(redis_listener(org_id))
    try:
        while True:
            # Keep connection alive; client can send pings
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket, org_id)
        listener_task.cancel()
