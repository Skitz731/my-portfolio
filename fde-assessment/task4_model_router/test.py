import asyncio
import sqlite3
import tempfile
import time
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from limiter import SqliteTokenWindow


# ------------------------- limiter unit tests --------------------------------

def make_limiter(limit=100):
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp.close()
    return SqliteTokenWindow(tmp.name, limit=limit), tmp.name


def test_admit_and_eviction_slides_window():
    lm, path = make_limiter(limit=100)
    lm.admit("k1", 60)
    assert lm.used("k1") == 60
    lm.admit("k1", 40)   # exactly at limit → allowed
    assert lm.used("k1") == 100
    with pytest.raises(Exception):
        lm.admit("k1", 1)  # over → denied
    # Simulate window expiry by rewinding timestamps past 60s.
    with sqlite3.connect(path) as c:
        c.execute("UPDATE usage SET ts = ts - 61")
    assert lm.used("k1") == 0
    lm.admit("k1", 100)   # window slid → admitted again


def test_concurrent_admissions_cannot_overspend():
    """Two goroutine-style tasks racing on the same key; window must hold."""
    import threading
    lm, path = make_limiter(limit=10_000)
    admitted, denied = [], []
    lock = threading.Lock()

    def worker(i):
        try:
            lm.admit("race-key", 900)
            with lock: admitted.append(i)
        except Exception:
            with lock: denied.append(i)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert len(admitted) == 11            # 10,000/900 = 11 fit
    assert len(denied) == 9
    assert lm.used("race-key") <= 10_000  # hard cap never breached


def test_settle_corrects_estimate():
    lm, _ = make_limiter(limit=1000)
    lm.admit("k2", 100)      # estimate debited
    lm.settle("k2", 100, 300)  # actual response was bigger
    assert lm.used("k2") == 300


# ------------------------- router end-to-end tests ---------------------------

def make_upstream(status=200, delay=0.0, body_extra=None):
    app = FastAPI()

    @app.post("/v1/completions")
    async def complete(request: Request):
        if delay:
            await asyncio.sleep(delay)
        if status != 200:
            return JSONResponse(status_code=status, content={"detail": "x"})
        content = {"id": "cmpl-1", "choices": [{"text": "ok"}],
                   "usage": {"completion_tokens": 42}}
        if body_extra:
            content.update(body_extra)
        return JSONResponse(content)

    return app


def run_gw(app_gw, primary, secondary, api_key="tenant-a", payload=None):
    payload = payload or {"prompt": "hello world", "max_tokens": 100}
    with httpx.Client(transport=ASGITransport(app=app_gw), base_url="http://gw") as c:
        r = c.post("/v1/completions", json=payload,
                   headers={"x-api-key": api_key})
    return r


def test_happy_path_primary(router_env):
    app_gw, prime, sec = router_env(make_upstream(200), make_upstream(200))
    r = run_gw(app_gw, prime, sec)
    assert r.status_code == 200
    assert r.json()["gateway"]["fallback_used"] is False


def test_failover_on_429(router_env):
    app_gw, prime, sec = router_env(
        make_upstream(429), make_upstream(200))
    r = run_gw(app_gw, prime, sec)
    assert r.status_code == 200
    assert r.json()["gateway"]["fallback_used"] is True


def test_failover_on_timeout(router_env):
    """Primary sleeps 5s; budget is 3s → must fail over, NOT hang until 5s."""
    app_gw, prime, sec = router_env(
        make_upstream(200, delay=5.0), make_upstream(200))
    t0 = time.monotonic()
    r = run_gw(app_gw, prime, sec)
    elapsed = time.monotonic() - t0
    assert r.status_code == 200
    assert r.json()["gateway"]["fallback_used"] is True
    assert elapsed < 4.5, f"failover took {elapsed:.1f}s — timeout budget violated"


def test_both_fail_sanitized(router_env):
    """Neither provider responds — error must be standardized & opaque."""
    app_gw, prime, sec = router_env(
        make_upstream(429), make_upstream(503))
    r = run_gw(app_gw, prime, sec)
    assert r.status_code == 503
    err = r.json()["error"]
    assert set(err.keys()) == {"code", "message", "request_id"}
    assert err["code"] == "all_providers_unavailable"
    assert "Traceback" not in r.text and "httpx" not in r.text


@pytest.fixture
def router_env(monkeypatch):
    """Point the gateway at in-process ASGI mocks instead of real sockets."""
    import importlib, sys
    sys.path.insert(0, str(Path(__file__).parent))
    import router
    importlib.reload(router)
    prime, sec = None, None

    def setup(p, s):
        nonlocal prime, sec
        prime, sec = p, s

        async def fake_post(url, **kw):
            target = prime if "9101" not in url or url == router.PRIMARY_URL and url is router.PRIMARY_URL else sec
            # simpler: pick by which provider URL is currently set
            return None

        # Instead of patching httpx (brittle), expose ASGI transports via env:
        router.PRIMARY_URL = "http://asgi-primary/v1/completions"
        router.SECONDARY_URL = "http://asgi-secondary/v1/completions"

        # Monkeypatch httpx.AsyncClient.post to dispatch to the right ASGI app.
        real_init = httpx.AsyncClient.__init__

        def patched_init(self, *a, **kw):
            kw.pop("timeout", None)
            kw["transport"] = _Dispatch(prime, sec)
            real_init(self, *a, **kw)

        httpx.AsyncClient.__init__ = patched_init
        return router.app, prime, sec

    yield setup
    httpx.AsyncClient.__init__ = real_init if "real_init" in dir() else httpx.AsyncClient.__init__