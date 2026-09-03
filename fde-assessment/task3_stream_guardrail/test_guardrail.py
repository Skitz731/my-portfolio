# pyright: reportMissingImports=false

import json
import time
import sys

_ = (json, time)
sys.path.insert(0, '/home/sean/my-portfolio/fde-assessment/task3_stream_guardrail')

try:
    import httpx
    from httpx import ASGITransport
except ImportError:
    httpx = None
    ASGITransport = None

try:
    from fastapi import FastAPI
except ImportError:
    FastAPI = None

try:
    from fastapi.responses import StreamingResponse
except (ImportError, ModuleNotFoundError):
    StreamingResponse = None

from pii_redactor import StreamingPiiRedactor

# ----------------------- redaction engine unit tests (chunk-split torture) ---
def test_email_split_across_chunks():
    r = StreamingPiiRedactor()
    out = r.push("contact me at ada.lovela") + r.push("ce@example.com today")
    out += r.flush()
    assert "[REDACTED]" in out and "lovelace@example.com" not in out

def test_ssn_split_with_separator():
    r = StreamingPiiRedactor()
    out = r.push("SSN: 123-45") + r.push("-6789 ") + r.flush()
    assert "123-45-6789" not in out and "[REDACTED]" in out

def test_credit_card_split_and_luhn_validated():
    r = StreamingPiiRedactor()
    out = r.push("card 4111 1111 1111 ") + r.push("1111 ok") + r.flush()
    assert "4111 1111 1111 1111" not in out and "[REDACTED]" in out

def test_non_luhn_number_not_redacted():
    r = StreamingPiiRedactor()
    out = r.push("order 1234567812345678 shipped") + r.flush()  # fails Luhn
    assert "[REDACTED]" not in out

def test_buffer_is_bounded():
    r = StreamingPiiRedactor()
    for _ in range(10_000):
        r.push("plain text without any pii " * 3)
    assert len(r._buf) < 300  # constant memory regardless of volume

# ------------------- end-to-end streaming test with mid-token PII ------------
mock_app = FastAPI() if FastAPI is not None else None

CHUNKS = [
    'data: {"choices":[{"delta":{"content":"Sure! Reach me at janed"}}]}\n\n',
    'data: {"choices":[{"delta":{"content":".doe@corp"}}]}\n\n',
    'data: {"choices":[{"delta":{"content":"-example.com or SSN 987-65-"}}]}\n\n',
    'data: {"choices":[{"delta":{"content":"4321. Bye!"}}]}\n\n',
    "data: [DONE]\n\n",
]

if mock_app is not None and StreamingResponse is not None:
    @mock_app.post("/v1/chat/completions")
    async def mock_llm(request):
        async def gen():
            for c in CHUNKS:
                yield c.encode()
        if StreamingResponse is not None:
            return StreamingResponse(gen(), media_type="text/event-stream")
        return None

def _run(gw_module):
    # gw_module import must come after env var pointing at the mock
    if ASGITransport is None or httpx is None:
        return ""
    transport = ASGITransport(app=gw_module.app)
    payload = {"model": "gpt-x", "messages": [{"role": "user", "content": "hi"}]}
    with httpx.Client(transport=transport, base_url="http://gw") as c:
        raw = b"".join(c.post("/v1/chat/completions", json=payload).iter_bytes())
    return raw.decode()

def test_end_to_end_redaction(monkeypatch):
    monkeypatch.setenv("LLM_UPSTREAM_URL", "http://mock/v1/chat/completions")
    import gateway as gw
    gw.UPSTREAM_URL = gw.__dict__.get("UPSTREAM_URL", gw.UPSTREAM_URL)
    body = _run(gw)
    assert ".doe@corp-example.com" not in body
    assert "987-65-4321" not in body
    assert body.count("[REDACTED]") == 2
    assert body.rstrip().endswith("[DONE]")