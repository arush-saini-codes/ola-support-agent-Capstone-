"""Pydantic request and response schemas for FastAPI endpoints (Section 15.1)."""

from typing import Literal
from pydantic import BaseModel, Field
from crew.schemas import TicketInfo, PipelineResult


class AskRequest(BaseModel):
    session_id: str = "default"
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    trace_id: str
    status: Literal["ok", "refused", "blocked", "rejected", "error"]
    answer: str
    intent: str
    sources: list[str] = Field(default_factory=list)
    ticket: TicketInfo | None = None
    approved: bool | None = None
    review_reason: str | None = None
    cache_hit: bool = False
    guardrail_flags: list[str] = Field(default_factory=list)
    tokens_used: int = 0
    cost_inr: float = 0.0
    latency_ms: float = 0.0

    @classmethod
    def from_pipeline_result(cls, res: PipelineResult) -> "AskResponse":
        return cls(
            trace_id=res.trace_id,
            status=res.status,
            answer=res.answer,
            intent=res.intent,
            sources=res.sources,
            ticket=res.ticket,
            approved=res.approved,
            review_reason=res.review_reason,
            cache_hit=res.cache_hit,
            guardrail_flags=res.guardrail_flags,
            tokens_used=res.tokens_used,
            cost_inr=res.cost_inr,
            latency_ms=res.latency_ms,
        )


class AddDocumentRequest(BaseModel):
    doc_id: str = Field(pattern=r"^[a-z0-9_]+$")
    title: str
    text: str = Field(min_length=20)


class AddDocumentResponse(BaseModel):
    doc_id: str
    chunks_added: int
    collection: str
    cache_cleared: bool


class HealthResponse(BaseModel):
    status: str = "ok"
    mock_llm: bool = True

