"""
Resilient model router: rate limiting, 3000ms primary timeout, automatic
failover to a secondary provider on 429/timeout/connection error, and
standardized sanitized error payloads (no upstream stack traces leak).
"""

import asyncio
import logging
import os
import sqlite3
import sys
import uuid
from pathlib import Path

try:
    import httpx  # type: ignore[import-not-found]
except ImportError as e:
    raise ImportError("httpx is required. Install it with: pip install httpx") from e

try:
    from fastapi import FastAPI, Request  # type: ignore[import-not-found]
    from fastapi.responses import JSONResponse  # type: ignore[import-not-found]
except ImportError as e:
    raise ImportError("fastapi is required. Install it with: pip install fastapi") from e

from limiter import (RateLimitExceeded, SqliteTokenWindow,
                     estimate_request_tokens)

logging.basicConfig(stream=sys.stderr, level=logging.INFO)  # stderr only
log = logging.getLogger("model-router")

PRIMARY_TIMEOUT_MS = 3000
PRIMARY_TIMEOUT_S = PRIMARY_TIMEOUT_MS / 1000

DB_PATH = Path(__file__).parent / "gateway.sqlite"

app = FastAPI(title="LLM Gateway — Rate Limiting & Failover Router")

limiter = SqliteTokenWindow(DB_PATH)

# Provider endpoints — mock-friendly via env override in tests.
PRIMARY_URL = os.environ.get("PRIMARY_URL", "http://127.0.0.1:9101/v1/completions")
SECONDARY_URL = os.environ.get("SECONDARY_URL", "http://127.0.0.1:9102/v1/completions")


def gateway_error(req_id: str, code: str, message: str, status: int) -> JSONResponse:
    """
    The ONLY error shape that leaves this gateway. Deliberately flat and
    opaque: codes are gateway-level, messages are human strings we author —
    raw upstream exceptions never pass through.
    """
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": req_id,
            }
        },
    )


async def call_provider(url: str, body: dict) -> tuple[int, dict | None, str | None]:
    """
    Returns (status_code, parsed_json_or_None, sanitized_failure_or_None).
    Raises nothing — all failure modes are converted into the tuple so the
    router's decision logic stays flat.
    """
    try:
        async with httpx.AsyncClient(timeout=PRIMARY_TIMEOUT_S) as client:
            resp = await client.post(url, json=body)
        if resp.status_code == 429:
            return 429, None, "rate_limited_upstream"
        if resp.status_code >= 500:
            return resp.status_code, None, "upstream_unavailable"
        try:
            return resp.status_code, resp.json(), None
        except Exception:
            return resp.status_code, None, "malformed_upstream_body"
    except asyncio.TimeoutError:
        # Important nuance: httpx wraps its timeout as httpx.TimeoutException,
        # which asyncio.wait_for also converts. Catch broadly but sanitize.
        return 0, None, "primary_timeout"
    except (httpx.TimeoutException, httpx.ConnectError, httpx.RequestError):
        return 0, None, "primary_unreachable"


@app.post("/v1/completions")
async def completions(request: Request) -> JSONResponse:
    req_id = uuid.uuid4().hex[:12]

    # ---------------- auth-ish: API key from header -----------------------
    api_key = request.headers.get("x-api-key", "")
    if not api_key:
        return gateway_error(req_id, "missing_api_key", "x-api-key header required", 401)

    try:
        body = await request.json()
    except Exception:
        return gateway_error(req_id, "invalid_request", "Request body must be JSON", 400)

    # ---------------- admission control -------------------------------------
    est = estimate_request_tokens(body)
    try:
        limiter.admit(api_key, est)
    except RateLimitExceeded as e:
        retry_after = max(1, int(60 * (e.used + e.requested - e.limit) / max(e.requested, 1)))
        resp = gateway_error(
            req_id, "rate_limit_exceeded",
            f"Tenant limit of {e.limit} tokens/min exceeded "
            f"({e.used} used, {e.requested} requested)", 429)
        resp.headers["Retry-After"] = str(retry_after)
        return resp
    except sqlite3.Error:
        return gateway_error(req_id, "internal_error", "Admission control unavailable", 503)

    # ---------------- routing: primary, then failover -----------------------
    status, data, failure = await call_provider(PRIMARY_URL, body)
    used_fallback = False

    if failure is not None or status >= 500 or status == 429:
        log.warning("[%s] primary failed (%s) — failing over", req_id, failure or status)
        status2, data2, failure2 = await call_provider(SECONDARY_URL, body)
        used_fallback = True
        if failure2 is not None or status2 >= 400:
            # Both legs failed: sanitized, no upstream internals exposed.
            return gateway_error(
                req_id, "all_providers_unavailable",
                "No upstream model provider could serve this request", 503)
        status, data = status2, data2

    # ---------------- settle actual usage ------------------------------------
    resp_tokens = 0
    if isinstance(data, dict):
        resp_tokens = int(data.get("usage", {}).get("completion_tokens", 0)) \
            if isinstance(data.get("usage"), dict) else 0
    try:
        limiter.settle(api_key, est, resp_tokens)
    except sqlite3.Error:
        log.error("[%s] settlement failed (non-fatal)", req_id)

    if isinstance(data, dict):
        data.setdefault("gateway", {})["fallback_used"] = used_fallback
    return JSONResponse(status_code=200, content=data)


@app.on_event("shutdown")
async def cleanup() -> None:
    # WAL checkpoint so the on-disk db is clean for inspection after test runs.
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")