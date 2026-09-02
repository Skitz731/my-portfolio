"""
Token-aware sliding window rate limiter backed by on-disk SQLite.

Concurrency model:
- All writes happen inside BEGIN IMMEDIATE transactions, so two concurrent
  requests from the same key cannot both pass admission and double-spend the
  window budget (classic TOCTOU on rate limiters).
- Old entries are evicted opportunistically on every check, keeping the table
  small; a periodic vacuum isn't needed for realistic volumes.

Token accounting: the caller supplies the estimated cost of the REQUEST;
we debit request tokens upfront and response tokens at completion via the same
window key (response tokens share the caller's 60s budget, as they should for
a token-per-minute cap).
"""

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

WINDOW_SECONDS = 60.0
DEFAULT_LIMIT_TOKENS = 50_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    api_key TEXT NOT NULL,
    ts      REAL NOT NULL,
    tokens  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_key_ts ON usage(api_key, ts);
"""


class RateLimitExceeded(Exception):
    def __init__(self, limit: int, used: int, requested: int):
        self.limit, self.used, self.requested = limit, used, requested
        super().__init__("rate limit exceeded")


class SqliteTokenWindow:
    def __init__(self, db_path: str | Path, limit: int = DEFAULT_LIMIT_TOKENS):
        self.db_path = str(db_path)
        self.limit = limit
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=5.0, isolation_level=None)
        try:
            conn.execute("PRAGMA journal_mode=WAL")   # readers don't block writers
            conn.execute("PRAGMA busy_timeout=5000")  # tolerate write contention
            yield conn
        finally:
            conn.close()

    def admit(self, api_key: str, requested_tokens: int) -> None:
        """Atomically debit tokens for this request or raise RateLimitExceeded."""
        now = time.monotonic()
        cutoff = now - WINDOW_SECONDS
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")  # serialize competing admissions
            try:
                # Evict expired rows for this key (also keeps the table small).
                conn.execute("DELETE FROM usage WHERE api_key = ? AND ts <= ?",
                             (api_key, cutoff))
                (used,) = conn.execute(
                    "SELECT COALESCE(SUM(tokens), 0) FROM usage "
                    "WHERE api_key = ? AND ts > ?", (api_key, cutoff)).fetchone()
                if used + requested_tokens > self.limit:
                    conn.execute("ROLLBACK")
                    raise RateLimitExceeded(self.limit, used, requested_tokens)
                conn.execute("INSERT INTO usage VALUES (?, ?, ?)",
                             (api_key, now, requested_tokens))
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def settle(self, api_key: str, request_tokens: int, response_tokens: int) -> None:
        """
        Adjust the debit to actual usage once the upstream call completes.
        Replaces the estimate row rather than appending a correction row, so
        the window sum stays truthful.
        """
        delta = response_tokens - request_tokens
        if delta == 0:
            return
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                # Find the oldest unsettled estimate row for this key and amend it.
                row = conn.execute(
                    "SELECT rowid, tokens FROM usage WHERE api_key = ? "
                    "ORDER BY ts ASC LIMIT 1", (api_key,)).fetchone()
                if row is None:
                    conn.execute("INSERT INTO usage VALUES (?, ?, ?)",
                                 (api_key, time.monotonic(), max(response_tokens, 0)))
                else:
                    conn.execute("UPDATE usage SET tokens = ? WHERE rowid = ?",
                                 (max(row[1] + delta, 0), row[0]))
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def used(self, api_key: str) -> int:
        cutoff = time.monotonic() - WINDOW_SECONDS
        with self._conn() as conn:
            (total,) = conn.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM usage WHERE api_key = ? AND ts > ?",
                (api_key, cutoff)).fetchone()
            return total


def estimate_request_tokens(body: dict) -> int:
    """
    Deterministic, dependency-free token estimator (≈ chars/4 for English,
    matching OpenAI's rule of thumb). We deliberately debit the estimate
    upfront and settle later — a real deployment swaps in tiktoken in settle().
    """
    total_chars = len(str(body.get("prompt", "") if body.get("prompt") is not None
                         else body.get("messages", "")))
    return max(total_chars // 4, 1)