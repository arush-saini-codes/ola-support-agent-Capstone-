"""Per-request budget caps and token metering (Section 14.3).

Provides two layers of budget protection:
1. Pre-flight check: rejects oversized user input before any model, tool, or guardrail runs.
2. In-flight check: accumulates prompt and output tokens across CrewAI and AutoGen calls,
   raising BudgetExceeded immediately if MAX_REQUEST_TOKENS is crossed.
"""

import config  # Ground Rule 3: must be first import
import math
from contextvars import ContextVar


def estimate_tokens(text: str) -> int:
    """Heuristic token estimation: 1 token per 4 characters."""
    return math.ceil(len(text) / 4)


def compute_cost_inr(tokens: int) -> float:
    """Compute notional INR cost from token count."""
    return round((tokens / 1000.0) * config.COST_PER_1K_TOKENS_INR, 4)


class BudgetExceeded(Exception):
    """Raised when request tokens cross pre-flight or in-flight budget caps."""

    def __init__(
        self,
        stage: str,
        tokens: int,
        cap: int,
        cost_inr: float | None = None,
        refused_call: str = "",
    ):
        self.stage = stage
        self.tokens = tokens
        self.cap = cap
        self.refused_call = refused_call
        self.cost_inr = cost_inr if cost_inr is not None else compute_cost_inr(tokens)
        call_info = f" on '{refused_call}'" if refused_call else ""
        super().__init__(
            f"Budget exceeded at stage '{stage}'{call_info}: {tokens} tokens crossed cap of {cap} "
            f"(notional cost: Rs {self.cost_inr:.4f})"
        )


class TokenMeter:
    """Context-bound token meter enforcing in-flight budget caps."""

    def __init__(self, cap: int | None = None):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cap = cap if cap is not None else config.MAX_REQUEST_TOKENS

    @property
    def used(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def check_before_call(self, prompt_tokens: int, call_name: str = "", stage: str = "in_flight"):
        """Check whether prompt_tokens would exceed the cap BEFORE running the call.
        
        Does not charge the meter. Raises BudgetExceeded without making or charging the call.
        """
        if self.cap is not None and (self.used + prompt_tokens) > self.cap:
            raise BudgetExceeded(
                stage=stage,
                tokens=self.used,
                cap=self.cap,
                cost_inr=compute_cost_inr(self.used),
                refused_call=call_name,
            )

    def add(self, prompt: int = 0, completion: int = 0):
        """Charge tokens after a call succeeds."""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        total = self.total_tokens

        if self.cap is not None and total > self.cap:
            raise BudgetExceeded(
                stage="in_flight",
                tokens=total,
                cap=self.cap,
                cost_inr=compute_cost_inr(total),
            )


_REQUEST_TOKEN_METER: ContextVar[TokenMeter | None] = ContextVar("request_token_meter", default=None)


def get_token_meter() -> TokenMeter:
    """Return the active TokenMeter for the current context, creating one if needed."""
    meter = _REQUEST_TOKEN_METER.get()
    if meter is None:
        meter = TokenMeter()
        _REQUEST_TOKEN_METER.set(meter)
    return meter


def reset_token_meter(cap: int | None = None) -> TokenMeter:
    """Reset and return a fresh TokenMeter for the current context."""
    meter = TokenMeter(cap=cap)
    _REQUEST_TOKEN_METER.set(meter)
    return meter

