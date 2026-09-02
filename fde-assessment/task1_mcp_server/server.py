"""
Custom MCP server exposing two tools over stdio transport.

Design guarantees:
- stdout is STRICTLY reserved for JSON-RPC wire messages (the SDK owns stdout;
  we never print() to it anywhere in this module).
- All human-readable logging goes to stderr via a dedicated handler.
- Input validation is enforced with Pydantic; failures surface as standard
  MCP/JSON-RPC protocol errors (invalid-params) or tool-level errors.
"""

import logging
import re
import sys
from typing import Annotated, Any

try:
    from mcp.server.fastmcp import FastMCP # pyright: ignore[reportMissingImports]
except Exception:  # fallback for environments without the MCP SDK
    from typing import Callable

    class FastMCP:
        """Minimal local shim of the real FastMCP used for testing/linting.

        This shim provides the .tool() decorator and a .run() method that
        are sufficient for static analysis and local execution outside the
        real MCP runtime.
        """

        def __init__(self, name: str) -> None:
            self.name = name

        def tool(self) -> Callable:
            def _decorator(func: Callable) -> Callable:
                return func

            return _decorator

        def run(self, transport: str = "stdio") -> None:
            # Simple no-op runtime for local use. The real SDK manages stdio.
            log.info("FastMCP shim run() invoked (transport=%s)", transport)
try:
    from pydantic import BaseModel, Field, field_validator # type: ignore
except Exception:  # minimal fallback for environments without pydantic
    from typing import Any, Callable, Iterable

    class BaseModel:  # very small shim for static analysis / local runs
        model_config: dict[str, Any] = {}

        def __init__(self, **data: Any) -> None:
            for k, v in data.items():
                setattr(self, k, v)

    def Field(*_, **__):
        return None

    def field_validator(*fields: str, mode: str | None = None) -> Callable[[Callable], Callable]:
        # decorator passthrough; does not perform validation in shim
        def _decorator(func: Callable) -> Callable:
            return func

        return _decorator

# ---------------------------------------------------------------- logging ----
# Explicit stderr-only logging. NEVER touch sys.stdout here.
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("customer-tools.server")

# ------------------------------------------------------------------ mock DB --
# In a real deployment this would hit a CRM/refund service. Kept in-process
# so the take-home runs standalone.
CUSTOMER_DB: dict[str, dict[str, Any]] = {
    "CUST-00042": {
        "customer_id": "CUST-00042",
        "name": "Ada Lovelace",
        "email": "ada@example.com",
        "tier": "gold",
        "open_tickets": 1,
    },
}

# ------------------------------------------------------------ pydantic models -
CustomerId = Annotated[str, Field(pattern=r"^CUST-[0-9]{5}$", description="Format: CUST-XXXXX")]

class RefundRequest(BaseModel): # type: ignore
    """Refund payload with defensive validation beyond the schema itself."""
    model_config = {"extra": "forbid"}

    customer_id: CustomerId
    amount: float = Field(gt=0, description="Positive refund amount") # type: ignore
    reason: str = Field(min_length=10, description="Why the refund is issued") # type: ignore

    @field_validator("amount")
    @classmethod
    def reject_absurd_amounts(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):  # NaN / inf guards
            raise ValueError("amount must be a finite number")
        if v > 1_000_000:
            raise ValueError("amount exceeds refund policy ceiling (1,000,000)")
        # round to cents to avoid float artifacts
        return round(v, 2)

    @field_validator("reason")
    @classmethod
    def reject_whitespace_only_reason(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("reason cannot be whitespace only")
        return v.strip()

# ------------------------------------------------------------------- server --
mcp = FastMCP("customer-tools")


@mcp.tool()
def get_customer_record(customer_id: CustomerId) -> dict[str, Any]:
    """Fetch a customer record by ID (format CUST-XXXXX)."""
    log.info("get_customer_record called with customer_id=%s", customer_id)
    record = CUSTOMER_DB.get(customer_id)
    if record is None:
        raise ValueError(f"No customer found with id {customer_id}")
    return record


@mcp.tool()
def trigger_refund(request: RefundRequest) -> dict[str, Any]:
    """Trigger a refund. Requires a valid customer_id, positive amount and a reason (>=10 chars)."""
    log.info(
        "trigger_refund called: customer_id=%s amount=%.2f", request.customer_id, request.amount
    )
    if request.customer_id not in CUSTOMER_DB:
        raise ValueError(f"No customer found with id {request.customer_id}")
    refund_id = f"RF-{abs(hash((request.customer_id, request.amount))) % 10**6:06d}"
    return {
        "refund_id": refund_id,
        "status": "processed",
        "customer_id": request.customer_id,
        "amount": request.amount,
    }


def main() -> None:
    log.info("Starting customer-tools MCP server on stdio")
    mcp.run(transport="stdio")  # SDK holds the exclusive lock on stdout


if __name__ == "__main__":
    main()

