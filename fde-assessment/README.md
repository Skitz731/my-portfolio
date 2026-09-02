# FDE Assessment — MCP Servers, Gateways & LLM Guardrails (Python)

Four runnable deliverables sharing one consistent stack: `fastapi`, `httpx`,
`pydantic`, the official `mcp` SDK, and on-disk SQLite for rate limiting.

## Setup
    pip install -e ".[dev]"
    pytest                    # runs all suites for tasks 1–4

## Task 1 — MCP server (strict validation, stdio)
    python -m task1_mcp_server.server            # speaks JSON-RPC on stdio
    pytest task1_mcp_server -v                   # incl. stdout-purity test
Notes: stdout is exclusively JSON-RPC; all logging goes to stderr.
Pydantic enforces `CUST-XXXXX`, positive finite amounts, reason ≥ 10 chars.

## Task 2 — MCP security gateway (role-based tool filtering)
    uvicorn task2_gateway_proxy.downstream_mock:app --port 9000 &
    uvicorn task2_gateway_proxy.gateway:app --port 8000 &
Tokens: `Bearer admin:supersecret-admin` / `Bearer viewer:read-only-secret`.
`admin_*` tool calls from non-admins are intercepted locally with
JSON-RPC `-32001` — the downstream server is never contacted.

## Task 3 — LLM gateway streaming PII redaction
    uvicorn task3_stream_guardrail.gateway:app --port 8000 &
Point `LLM_UPSTREAM_URL` at any OpenAI-compatible SSE endpoint.
Design: bounded tail-hold buffer (HOLD = max pattern length) — a match is
only missable if longer than HOLD, so redaction is lossless while memory
stays O(HOLD) and TTFT is one chunk.

## Task 4 — token-aware rate limiter + model failover router
    uvicorn task4_model_router.router:app --port 8000 &
Send `x-api-key` + JSON body to `POST /v1/completions`.
Sliding 60 s / 50,000-token window in SQLite (WAL, BEGIN IMMEDIATE),
3000 ms primary timeout, automatic failover on 429/timeout/5xx, and a
single sanitized `{code, message, request_id}` error shape.

## Assessment-wide design stance
- stdout purity (T1) and stderr-only logging (T2–T4) as deploy hygiene
- authorization decided *before* any downstream I/O (T2)
- lossless-over-latency trade for PII: hold ≤ 200 chars, never leak (T3)
- atomic admission control — concurrency cannot overspend the window (T4)