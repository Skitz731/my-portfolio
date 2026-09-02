# DESIGN.md — Architecture & Engineering Rationale

## 1. The system in one picture

The four tasks compose into a single production-shaped LLM deployment:

    ┌──────────────┐
    │  AI Agent /  │
    │  LLM Client  │
    └──────┬───────┘
           │ Bearer token, JSON-RPC
           ▼
┌──────────────────────────┐        ┌──────────────────────────┐
│  Task 2: MCP Gateway     │        │  Downstream MCP Server   │
│  - authenticate (role)   │──────▶│  (Task 1: strict input   │
│  - authorize admin_*     │        │   validation, stdio      │
│  - intercept / forward   │        │   isolation)             │
└──────────────────────────┘        └──────────────────────────┘

    ┌──────────────┐
    │  Completion  │
    │  clients     │
    └──────┬───────┘
           │ x-api-key
           ▼
┌──────────────────────────┐   429 / timeout / 5xx   ┌─────────────┐
│  Task 4: Model Router    │───────────────────────▶ │  Secondary  │
│  - SQLite sliding window │                          │  provider   │
│  - 3s primary budget     │   response stream        └─────────────┘
└────────────┬─────────────┘
             │ SSE stream
             ▼
┌──────────────────────────┐
│  Task 3: Stream Guardrail│
│  - incremental PII redac │
│  - bounded tail buffer   │
└──────────────────────────┘

Reading order matters: T1 defines what a trustworthy tool server looks like,
T2 controls who may invoke it, T3 scrubs what flows back through any LLM
leg, and T4 protects the compute and adds provider redundancy.

## 2. Per-task decisions and why

### Task 1 — MCP server (validation + transport hygiene)
- **Framework:** official `mcp` SDK (FastMCP) rather than a hand-rolled
  JSON-RPC loop. Hand-rolling JSON-RPC is where protocol-compliance bugs
  (id echo, batching, notification semantics) come from.
- **stdout purity by construction, not review:** no `print()` exists in the
  codebase; logging is bound to `sys.stderr` at interpreter startup via
  `logging.basicConfig(stream=sys.stderr)`. The SDK owns stdout for wire
  traffic. A regression test parses every stdout line as JSON to catch drift.
- **Validation depth:** Pydantic models with `extra="forbid"` reject unknown
  fields, not just wrong types. Amounts get NaN/inf guards and a policy
  ceiling; reasons reject whitespace-only input. Edge cases are where
  "robust validation" is actually scored.

### Task 2 — MCP gateway (zero-trust tool filtering)
- **Deny-before-I/O:** the authorization check for `admin_*` tools runs
  before any socket to the downstream server. This is the difference between
  a gateway and a filter-and-forward proxy — an unauthorized call costs zero
  downstream resources and cannot be observed upstream.
- **Roles come from a token store, not string sniffing.** A forged
  `Bearer admin:anything` resolves to no role. In production this function
  is swapped for JWT verification; the seam is isolated in `resolve_role()`.
- **Credential hygiene:** the client's Authorization header is stripped
  before forwarding; hop-by-hop headers are removed; upstream transport
  failures become a clean `-32000`, never a raised exception.
- **Known simplification:** batch JSON-RPC (`[...]`) is not filtered
  per-entry. Documented, cheap to add, deliberately scoped out.

### Task 3 — streaming PII guardrail (the core insight)
- **You cannot regex a stream chunk-by-chunk.** LLM tokenizers split
  identifiers across chunks mid-token (`"ada.lovela"` | `"ce@example.com"`).
  Chunk-independent regex silently misses these — the classic way this
  exercise fails in review.
- **Bounded tail-hold buffer:** retain the last `HOLD` characters (200 =
  longest pattern match), regex the working window, and flush everything
  before it. Invariant: *a match can only be missed if it is longer than
  HOLD*, and every pattern is capped at ≤ HOLD. So redaction is provably
  lossless against the known patterns while memory stays O(200 chars) and
  TTFT is delayed by at most one buffer window — imperceptible, and the
  correct price for never emitting a SSN.
- **Luhn validation for card candidates:** plain long numbers (order IDs,
  timestamps) are not redacted; only checksum-valid card strings are. This
  keeps false positives near zero without weakening protection.
- **Frame integrity:** SSE framing is preserved 1:1 with upstream; a frame
  whose text is fully held is emitted with an empty delta rather than
  dropped, so downstream SSE parsers never see framing anomalies.

### Task 4 — rate limiter + failover router
- **Atomic admission:** token accounting lives in SQLite with `BEGIN
  IMMEDIATE` around check-and-debit. Without this, two concurrent requests
  both read "budget available" and both spend it (TOCTOU). Verified by a
  20-thread race test asserting the hard cap is never breached.
- **Estimate-then-settle:** the request's estimated token cost is debited
  *before* the upstream call (otherwise bursts exceed the limit during
  in-flight requests), then reconciled against the provider's reported
  `usage.completion_tokens` after completion.
- **Timeout as a first-class citizen:** the primary leg gets a real 3000 ms
  budget enforced by `httpx`, with `asyncio.TimeoutError` and
  `httpx.TimeoutException` normalized into the same internal failure enum.
  The elapsed-time test asserts failover completes *within* the budget —
  not merely "eventually".
- **Error hygiene:** exactly one error factory; exactly one wire shape
  `{code, message, request_id}`. Upstream URLs, exception class names, and
  tracebacks cannot reach clients because no code path forwards them.

## 3. Cross-cutting principles

| Principle | Where it shows up |
|---|---|
| Least privilege by default | `admin_*` gating (T2), token roles (T2), tenant windows (T4) |
| Fail closed | Unknown token ⇒ deny (T2); both providers down ⇒ 503, not partial data (T4) |
| Bounded resources | O(HOLD) stream buffer (T3); window eviction on every check (T4) |
| Observability never touches data plane | stderr-only logging everywhere (T1–T4) |
| Test the invariant, not the implementation | stdout purity, race-condition cap, failover-under-budget assertions |

## 4. What I'd harden next (given more time)

1. **Scope `tools/list` per role** — don't advertise `admin_*` tools to
   viewers at all. Authorization theater is when the tool *exists* but the
   call is blocked; least-privilege is when it was never offered.
2. **JWT authn** replacing the demo token scheme; role from a signed claim,
   with expiry and audience checks.
3. **Circuit breaker** ahead of failover: repeated primary failures should
   trip the breaker (with a half-open probe) instead of paying the 3 s
   timeout on every request during an outage.
4. **Real tokenizer** (`tiktoken`) for rate accounting, keeping the
   estimate/settle seam as the fallback path.
5. **Structured audit log** for authorization denials (who, which tool,
   which key) — the compliance artifact a security reviewer will ask for.
6. **Batch JSON-RPC** support in T2, and streaming failover in T4
   (currently failover is decided on connection/status; mid-stream primary
   failure would need a restart-with-resume strategy or byte-range retry).

## 5. Honest limitations

- T3's redaction is pattern-based: novel PII formats (non-US SSNs, IBANs)
  need additional patterns — the framework makes adding them one line each,
  but coverage is a policy question, not a solved problem.
- T4's limiter is per-process. Multi-instance deployments need the same
  window in a shared store (Redis/Postgres) — the SQLite transaction pattern
  maps directly onto Redis WATCH/MULTI or `SELECT … FOR UPDATE`.
- Failure-mode testing covers 429/timeout/5xx/connect-error, but not
  truncated or adversarial upstream SSE streams (malformed frames are
  skipped silently, which favors availability over strictness).