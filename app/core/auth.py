"""Shared-secret gate for a deployed backend.

Locally there is nothing to protect: the server listens on 127.0.0.1 and
only you can reach it. The moment it is on a public host that stops being
true -- the websockets take audio and spend your Groq/Anthropic credit, so
an open URL is an open tab on your API bill.

Set APP_AUTH_TOKEN on the server and every client has to present the same
string. Leave it unset (the default) and nothing changes, so local
development and the offline test clients keep working untouched.

Clients present it as:
  - HTTP: `Authorization: Bearer <token>` or `X-Auth-Token: <token>`
  - WebSocket: a `token=<token>` query parameter (browsers cannot set
    headers on a websocket handshake, and it keeps the desktop clients to
    one code path)
"""

import hmac

from fastapi import HTTPException, Request, WebSocket

from app.core.config import get_settings


def _presented_http_token(request: Request) -> str | None:
    header = request.headers.get("authorization")
    if header and header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.headers.get("x-auth-token") or request.query_params.get("token")


def _matches(presented: str | None, expected: str) -> bool:
    # compare_digest rather than == so a wrong token cannot be recovered
    # one character at a time by timing the response.
    return bool(presented) and hmac.compare_digest(presented, expected)


async def require_token(request: Request) -> None:
    """FastAPI dependency for the HTTP routes."""
    expected = get_settings().app_auth_token
    if not expected:
        return
    if not _matches(_presented_http_token(request), expected):
        raise HTTPException(status_code=401, detail="Invalid or missing auth token.")


async def websocket_token_ok(websocket: WebSocket) -> bool:
    """Checks a websocket handshake. Returns False having already closed
    the socket, so callers can just `return`.

    The close happens BEFORE accept(), which makes it a rejected handshake
    (HTTP 403) rather than a connection that opens and then hangs up --
    clients see a clear failure instead of a mystery disconnect loop.
    """
    expected = get_settings().app_auth_token
    if not expected:
        return True

    presented = websocket.query_params.get("token")
    if not presented:
        header = websocket.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            presented = header[7:].strip()
        else:
            presented = websocket.headers.get("x-auth-token")

    if _matches(presented, expected):
        return True

    await websocket.close(code=1008)
    return False
