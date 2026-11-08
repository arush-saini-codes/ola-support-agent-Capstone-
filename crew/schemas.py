"""Pydantic schemas for structured output (Section 12.4).

Defines:
- TicketInfo: normalized structured ticket representation
- SupportResponse: strictly validated system response model
- PipelineResult: end-to-end pipeline execution result model
"""

from typing import Literal
from pydantic import BaseModel, Field


class TicketInfo(BaseModel):
    """Structured information about a queried support ticket."""
    record_id: str
    category: str
    severity: str
    status: str
    resolution_time_hours: float
    days_since_created: int
    escalated: bool
    escalation_score: float = Field(ge=0.0, le=1.0)
    escalation_recommended: bool


class SupportResponse(BaseModel):
    """The canonical validated response model produced by the crew."""
    answer: str = Field(min_length=1)
    intent: Literal["policy", "ticket", "mixed", "refused", "fallback", "blocked"]
    sources: list[str] = Field(default_factory=list)  # doc_ids
    ticket: TicketInfo | None = None
    grounded: bool
    top_similarity: float | None = None
    guardrail_flags: list[str] = Field(default_factory=list)


class PipelineResult(BaseModel):
    """End-to-end execution result returned by pipeline.answer()."""
    trace_id: str
    status: Literal["ok", "refused", "blocked", "rejected", "error"]
    answer: str
    intent: str
    masked_question: str = ""
    resolved_question: str | None = None
    sources: list[str] = Field(default_factory=list)
    rag_context: list[str] = Field(default_factory=list)
    ticket: TicketInfo | None = None
    grounded: bool
    top_similarity: float | None = None
    approved: bool | None = None
    review_reason: str | None = None
    cache_hit: bool = False
    guardrail_flags: list[str] = Field(default_factory=list)
    tokens_used: int = 0
    cost_inr: float = 0.0
    latency_ms: float = 0.0
    timings_ms: dict[str, float] = Field(default_factory=dict)
