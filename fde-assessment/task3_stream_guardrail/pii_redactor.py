import re

EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]{1,63})*\.[A-Za-z]{2,24}"
)
SSN_RE = re.compile(r"\b\d{3}[ -]\d{2}[ -]\d{4}\b")
CC_RE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")

REDACTED = "[REDACTED]"
MAX_MATCH_LEN = 200   # longest possible match across all patterns
HOLD = MAX_MATCH_LEN  # chars retained; everything older is flushed

_PATTERNS = (EMAIL_RE, SSN_RE, CC_RE)  # CC_RE defined below after Luhn helper

def _luhn_ok(candidate: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", candidate)]
    if not 13 <= len(digits) <= 19:
        return False
    total, parity = 0, (len(digits) - 2) % 2
    for i, d in enumerate(digits[:-1]):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return (total + digits[-1]) % 10 == 0

CC_RE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")

class StreamingPiiRedactor:
    def __init__(self) -> None:
        self._buf = ""

    def push(self, chunk: str) -> str:
        """Consume a chunk; return the prefix that is provably safe to emit now."""
        self._buf += chunk
        out: list[str] = []

        while True:
            best = None
            for pat in _PATTERNS:
                m = pat.search(self._buf)
                if m and (best is None or m.start() < best.start()):
                    best = m
            if best is None:
                break
            if hasattr(best, "re") and best.re is CC_RE and not _luhn_ok(best.group()):
                # Not a real card — mask only this candidate and rescan.
                self._buf = (self._buf[:best.start()]
                             + "\x00" * len(best.group())
                             + self._buf[best.end():])
                continue
            out.append(self._buf[:best.start()])
            out.append(REDACTED)
            self._buf = self._buf[best.end():]

        # Emit everything except the last HOLD chars (where a split match may hide).
        if len(self._buf) > HOLD:
            cut = len(self._buf) - HOLD
            out.append(self._buf[:cut])
            self._buf = self._buf[cut:]
        return "".join(out)

    def flush(self) -> str:
        """Call at stream end: redact any final match, drain the buffer."""
        for pat in _PATTERNS:
            self._buf = pat.sub(REDACTED, self._buf)
        tail, self._buf = self._buf, ""
        return tail