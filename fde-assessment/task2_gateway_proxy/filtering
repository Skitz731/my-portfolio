import logging
import sys
from typing import Any

try:
    import httpx
except Exception:  # pragma: no cover - fallback for environments without httpx
    class _RequestError(Exception):
        pass

    class _StubResponse:
        def __init__(self):
            self.status_code = 502

        def json(self):
            return {"jsonrpc": "2.0", "id": None,
                    "error": {"code": -32000, "message": "Upstream MCP server unavailable"}}

    class AsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            raise _RequestError("httpx not installed")

    # emulate the httpx namespace minimally
    class _httpx:
        RequestError = _RequestError
        AsyncClient = AsyncClient

    httpx = _httpx

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
except Exception:  # pragma: no cover - fallback for environments without fastapi
    # Minimal stubs so the module can be imported and tested without FastAPI
    class Request:  # very small subset used in this module
        def __init__(self, scope=None, receive=None):
            self._json = None

        async def json(self):
            return self._json or {}

    class JSONResponse:
        def __init__(self, content, status_code: int = 200, media_type: str = "application/json"):
            self.content = content
            self.status_code = status_code
            self.media_type = media_type

        # support ASGI response interface for tests that might inspect attributes
        def __call__(self):
            return self.content

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
log = logging.getLogger("mcp-gateway")

DOWNSTREAM_URL = "http://127.0.0.1:9000/mcp"
app = FastAPI(title="MCP Security Gateway")

TOKEN_STORE = {
    "admin:supersecret-admin": "admin",
    "viewer:read-only-secret": "viewer",
}

HOP_BY_HOP = {"connection", "keep-alive", "transfer-encoding", "upgrade", "te"}


def resolve_role(bearer_token: str | None) -> str | None:
    """Demo token scheme '<role>:<secret>'. Swap for JWT verification in prod."""
    if not bearer_token:
        return None
    token = bearer_token.strip()
    return TOKEN_STORE.get(token)


def tool_name_requires_admin(name: str) -> bool:
    return isinstance(name, str) and name.startswith("admin_")


def rpc_error(req_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id,
            "error": {"code": code, "message": message}}


def safe_json(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        return rpc_error(None, -32000, "Malformed upstream response")


@app.post("/mcp")
async def gateway(req: Request) -> JSONResponse:
    auth = req.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return JSONResponse(rpc_error(None, -32001, "Missing bearer token"),
                            status_code=401, media_type="application/json")

    role = resolve_role(auth[len("Bearer "):])
    if role is None:
        return JSONResponse(rpc_error(None, -32001, "Invalid or unknown token"),
                            status_code=401, media_type="application/json")

    try:
        payload = await req.json()
    except Exception:
        return JSONResponse(rpc_error(None, -32700, "Parse error"),
                            status_code=400, media_type="application/json")

    req_id = payload.get("id")
    method = payload.get("method")
    if not isinstance(method, str):
        return JSONResponse(rpc_error(req_id, -32600, "Invalid Request"),
                            status_code=400, media_type="application/json")

    if method == "tools/call":
        tool_name = (payload.get("params") or {}).get("name", "")
        if tool_name_requires_admin(tool_name) and role != "admin":
            log.warning("BLOCKED tools/call %r by role=%r", tool_name, role)
            # Intercept locally — never contact downstream.
            body = rpc_error(req_id, -32001, f"Unauthorized Tool Call: {tool_name}")
            return JSONResponse(body, status_code=200, media_type="application/json")

    headers = {k: v for k, v in req.headers.items()
               if k.lower() not in HOP_BY_HOP and k.lower() != "authorization"}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            upstream = await client.post(DOWNSTREAM_URL, json=payload, headers=headers)
    except httpx.RequestError as exc:
        log.error("Downstream failure: %s", type(exc).__name__)
        return JSONResponse(rpc_error(req_id, -32000, "Upstream MCP server unavailable"),
                            status_code=502, media_type="application/json")

    return JSONResponse(safe_json(upstream), status_code=upstream.status_code,
                        media_type="application/json")