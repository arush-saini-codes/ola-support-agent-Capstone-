"""Thread-safe counters and token meter for LLM/tool accounting (Section 11.1).

Tracks:
- crew_llm_calls
- review_llm_calls
- judge_calls
- generation_calls
- tool_calls
- TokenMeter (per-request token accounting in a context variable)
"""

import threading
from contextvars import ContextVar

_LOCK = threading.Lock()

_COUNTERS = {
    "crew_llm_calls": 0,
    "review_llm_calls": 0,
    "judge_calls": 0,
    "generation_calls": 0,
    "tool_calls": 0,
}

TOOL_CALL_LOG: list[dict] = []


def log_tool_call(tool: str, args: dict, agent: str = ""):
    """Record a tool invocation in TOOL_CALL_LOG (Section 10.4)."""
    with _LOCK:
        entry = {"tool": tool, "args": args}
        if agent:
            entry["agent"] = agent
        TOOL_CALL_LOG.append(entry)


def increment(name: str, by: int = 1) -> int:
    """Increment a named global counter in a thread-safe manner."""
    with _LOCK:
        if name not in _COUNTERS:
            _COUNTERS[name] = 0
        _COUNTERS[name] += by
        return _COUNTERS[name]


def get_counter(name: str) -> int:
    """Get the current value of a global counter."""
    with _LOCK:
        return _COUNTERS.get(name, 0)


def get_all_counters() -> dict[str, int]:
    """Get a copy of all current counters."""
    with _LOCK:
        return dict(_COUNTERS)


def reset_counters():
    """Reset all global counters to zero and clear tool call logs."""
    with _LOCK:
        for k in _COUNTERS:
            _COUNTERS[k] = 0
        TOOL_CALL_LOG.clear()


# TokenMeter imported from governance.budget for single source of truth
from governance.budget import TokenMeter, get_token_meter, reset_token_meter

