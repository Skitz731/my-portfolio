"""
LLM Gateway with real-time PII redaction on the response stream.

- Speaks OpenAI-compatible SSE on /v1/chat/completions.
- Upstream provider URL is configurable (defaults to a local mock).
- Bytes are decoded incrementally; SSE frames are re-emitted with redacted
  deltas. A frame whose entire delta is still inside the hold buffer is
  forwarded with an empty delta string, keeping framing 1:1 with upstream.
"""

import json
import logging
import os
import sys
from typing import AsyncIterator

try:
    # fastapi imports
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
except Exception as e:  # pragma: no cover - helpful error when deps missing
    raise RuntimeError(
        "fastapi is required to run this gateway. Install it with: pip install fastapi\n"
        f"Original error: {e}"
    )

try:
    import httpx
except Exception as e:  # pragma: no cover - helpful error when deps missing
    raise RuntimeError(
        "httpx is required to run this gateway. Install it with: pip install httpx\n"
        f"Original error: {e}"
    )

from pii_redactor import StreamingPiiRedactor

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("llm-gateway")

UPSTREAM_URL = os.environ.get("LLM_UPSTREAM_URL", "http://127.0.0.1:9001/v1/chat/completions")
app = FastAPI(title="LLM Gateway — streaming PII guardrail")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    redactor = StreamingPiiRedactor()

    async def relay() -> AsyncIterator[bytes]:
        client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=None))
        try:
            async with client.stream("POST", UPSTREAM_URL, json=body) as upstream:
                if upstream.status_code != 200:
                    yield _error_frame("upstream error", 502)
                    return
                async for line in _sse_lines(upstream.aiter_bytes()):
                    if not line.startswith("data:"):
                        yield (line + "\n\n").encode()   # framing/comments pass through
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        tail = redactor.flush()
                        if tail:
                            yield _delta_frame(tail)
                        yield "data: [DONE]\n\n"
                        return
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = _extract_delta(event)
                    safe = redactor.push(delta)
                    if safe or delta == "":
                        yield _delta_frame_with(safe, event)
                    # if safe == "" and delta != "": text is held; frame withheld.
        finally:
            await client.aclose()

    return StreamingResponse(
        relay_and_log(relay()),   # wrapper defined below
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )