# task2_gateway_proxy/test_gateway.py
import threading
import time

import httpx
import pytest
import uvicorn

from task2_gateway_proxy.gateway import app as gateway_app
from task2_gateway_proxy.downstream_mock import app as mock_app

ADMIN = {"Authorization": "Bearer admin:supersecret-admin"}
VIEWER = {"Authorization": "Bearer viewer:read-only-secret"}


@pytest.fixture(scope="session")
def downstream():
    config = uvicorn.Config(mock_app, host="127.0.0.1", port=9000, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # wait for readiness
    import time
    for _ in range(50):
        try:
            httpx.get("http://127.0.0.1:9000/docs")
            break
        except Exception:
            time.sleep(0.1)
    yield
    server.should_exit = True
    thread.join(timeout=5)


def client():
    return httpx.Client(transport=httpx.ASGITransport(app=gateway), base_url="http://gw")


def rpc(method, params=None, req_id=1):
    p = {"jsonrpc": "2.0", "method": method, "id": req_id}
    if params is not None:
        p["params"] = params
    return p


# --- 1. Wire-format parsing ---------------------------------------------------
def test_malformed_body_is_protocol_error(downstream):
    with client() as c:
        r = c.post("/mcp", headers=VIEWER, content=b"{not json",
                   headers_extra=None) if False else c.post(
            "/mcp", headers=VIEWER, content=b"{not json")
        assert r.status_code in (400,)
        assert r.json()["error"]["code"] == -32700


def test_missing_auth_is_rejected(downstream):
    with client() as c:
        r = c.post("/mcp", json=rpc("tools/list"))
        assert r.status_code == 401
        assert r.json()["error"]["code"] == -32001


# --- 2. Proxy forwarding ------------------------------------------------------
def test_tools_list_forwarded_transparently(downstream):
    with client() as c:
        r = c.post("/mcp", headers=VIEWER, json=rpc("tools/list"))
        body = r.json()
        names = {t["name"] for t in body["result"]["tools"]}
        assert {"admin_reset_key", "lookup_order"} <= names


# --- 3. Fine-grained authorization --------------------------------------------
def test_viewer_blocked_from_admin_tool(downstream):
    with client() as c:
        r = c.post("/mcp", headers=VIEWER, json=rpc(
            "tools/call", {"name": "admin_reset_key",
                           "arguments": {"key": "k"}}))
        body = r.json()
        assert body["error"]["code"] == -32001
        assert "Unauthorized" in body["error"]["message"]
        assert body["id"] == 1  # id echoed correctly in the error envelope


def test_viewer_can_call_non_admin_tool(downstream):
    with client() as c:
        r = c.post("/mcp", headers=VIEWER, json=rpc(
            "tools/call", {"name": "get_weather", "arguments": {"city": "Geneva"}}))
        assert "result" in r.json()


def test_admin_can_call_admin_tool(downstream):
    with client() as c:
        r = c.post("/mcp", headers=ADMIN, json=rpc(
            "tools/call", {"name": "admin_reset_key", "arguments": {"key": "k"}}))
        assert "result" in r.json()


def test_forged_admin_token_rejected(downstream):
    with client() as c:
        r = c.post("/mcp", headers={"Authorization": "Bearer admin:not-the-real-secret"},
                   json=rpc("tools/call", {"name": "admin_purge_cache", "arguments": {}}))
        assert r.json()["error"]["code"] == -32001