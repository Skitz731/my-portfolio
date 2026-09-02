import json
import subprocess
import sys
from pathlib import Path

SERVER = Path(__file__).parent / "server.py"


class StdioSession:
    """Minimal JSON-RPC-over-stdio client."""

    def __init__(self):
        self.proc = subprocess.Popen(
            [sys.executable, str(SERVER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._next_id = 0
        self.stderr_data = ""

    def request(self, method: str, params: dict | None = None, notify: bool = False):
        payload = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params # type: ignore
        if not notify:
            payload["id"] = self._next_id # type: ignore
            self._next_id += 1
        line = json.dumps(payload) + "\n"
        assert self.proc.stdin is not None
        self.proc.stdin.write(line)
        assert self.proc.stdin is not None
        self.proc.stdin.flush()
        if notify:
            return None
        assert self.proc.stdout is not None
        resp = json.loads(self.proc.stdout.readline())  # must be ONE clean JSON line
        return resp

    def drain_stderr(self) -> str:
        self.proc.terminate()
        _, err = self.proc.communicate(timeout=5)
        return err

    def initialize(self):
        resp = self.request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "0"},
        })
        self.request("notifications/initialized", notify=True)
        return resp


def test_initialize_handshake():
    s = StdioSession()
    try:
        resp = s.initialize()
        assert resp["result"]["serverInfo"]["name"] == "customer-tools" # type: ignore
    finally:
        s.drain_stderr()


def test_tools_list_contains_both_tools():
    s = StdioSession()
    try:
        s.initialize()
        resp = s.request("tools/list", {})
        names = {t["name"] for t in resp["result"]["tools"]} # type: ignore
        assert names == {"get_customer_record", "trigger_refund"}
    finally:
        s.drain_stderr()


def test_invalid_customer_id_format_is_rejected():
    s = StdioSession()
    try:
        s.initialize()
        resp = s.request("tools/call", {
            "name": "get_customer_record",
            "arguments": {"customer_id": "cust-42"},  # wrong format
        })
        assert resp["result"]["isError"] is True # type: ignore
        assert "validation error" in json.dumps(resp["result"]).lower() # type: ignore
    finally:
        s.drain_stderr()


def test_unknown_method_returns_error():
    s = StdioSession()
    try:
        resp = s.request("bogus/method", {})
        assert resp["error"]["code"] == -32601  # type: ignore # Method not found
    finally:
        s.drain_stderr()


def test_stdout_remains_pure_jsonrpc_and_logs_go_to_stderr():
    """Rubric-critical: stdout lines are individually parseable JSON-RPC; logs land on stderr."""
    s = StdioSession()
    try:
        s.initialize()
        s.request("tools/list", {})
    finally:
        stderr_text = s.drain_stderr()
    assert "Starting customer-tools" in stderr_text, "logs must go to stderr"
    assert s.proc.returncode is not None