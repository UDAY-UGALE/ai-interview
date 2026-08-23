from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.auth import websocket_token_ok
from app.services.answer_hub import answer_hub


router = APIRouter()


@router.websocket("/ws/answers")
async def answers_websocket(websocket: WebSocket) -> None:
    if not await websocket_token_ok(websocket):
        return

    session_id = websocket.query_params.get("session_id", "default")
    await websocket.accept()
    await answer_hub.connect(session_id, websocket)
    await websocket.send_json({"type": "ready", "session_id": session_id})

    try:
        while True:
            message = await websocket.receive_json()
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        await answer_hub.disconnect(session_id, websocket)
