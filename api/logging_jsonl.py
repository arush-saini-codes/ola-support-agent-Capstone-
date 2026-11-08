"""Structured JSONL logging for HTTP and WebSocket requests (Section 15.3).

Logs exactly one JSON line per request/turn to logs/requests.jsonl.
Thread-safe, UTF-8, ensure_ascii=False.
Ensures raw user text is never logged — request_text is always masked.
"""

import config  # Ground Rule 3: must be first import
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def get_current_iso_ts() -> str:
    """Return ISO 8601 UTC timestamp ending in Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def truncate_request_text(text: str, max_chars: int = 500) -> str:
    """Truncate request text to max_chars with a suffix denoting original length."""
    if len(text) > max_chars:
        return text[:max_chars] + f"…[truncated, {len(text)} chars]"
    return text


def log_request(record: dict[str, Any]) -> None:
    """Append a structured JSON line to logs/requests.jsonl."""
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = config.LOGS_DIR / "requests.jsonl"

    if "ts" not in record:
        record["ts"] = get_current_iso_ts()

    if "request_text" in record and isinstance(record["request_text"], str):
        record["request_text"] = truncate_request_text(record["request_text"])

    line = json.dumps(record, ensure_ascii=False)
    with _LOCK:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def log_ws_event(
    event: str,
    session_id: str,
    trace_id: str,
    turns_completed: int | None = None,
    latency_ms: float | None = None,
) -> None:
    """Log WebSocket connection lifecycle events (ws_connect, ws_disconnect)."""
    record: dict[str, Any] = {
        "ts": get_current_iso_ts(),
        "trace_id": trace_id,
        "channel": "ws",
        "endpoint": "/ws/chat",
        "session_id": session_id,
        "event": event,
    }
    if turns_completed is not None:
        record["turns_completed"] = turns_completed
    if latency_ms is not None:
        record["latency_ms"] = latency_ms
    log_request(record)

