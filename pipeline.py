"""The Single Entry Point Pipeline for Ola Support Agent (Sections 5 & 12.7).

Executes end-to-end request lifecycle:
1. Budget pre-flight check (Phase 8 stub)
2. Prompt injection detection
3. PII masking (phone numbers -> [PHONE])
4. Multi-turn session memory reference resolution
5. Response cache lookup (Phase 8 stub)
6. Intent routing
7. CrewAI multi-agent crew execution
8. SupportResponse schema validation
9. Output groundedness verification
10. Autogen review stage (Phase 7 stub)
11. Response cache store (Phase 8 stub)
12. Session memory turn append
"""

import config  # Ground Rule 3: must be first import
import time
import uuid
import asyncio
import concurrent.futures
from crew.schemas import PipelineResult, SupportResponse
from guardrails.pii import mask_pii
from guardrails.injection import detect_injection
from guardrails.groundedness import check_grounded
from memory.session import process_turn_memory, append_turn_to_history
from crew.crew import run_crew
from tools.ticket_tool import check_support_ticket_status
from rag.generate import grounded_answer
from review.autogen_review import run_review
from governance.budget import (
    estimate_tokens,
    compute_cost_inr,
    BudgetExceeded,
    reset_token_meter,
)
from governance.cache import cache


async def answer(
    session_id: str, question: str, request_cap: int | None = None
) -> PipelineResult:
    """Async single entry point for answering user queries."""
    t_start = time.perf_counter()
    trace_id = str(uuid.uuid4())
    timings_ms: dict[str, float] = {}
    guardrail_flags: list[str] = []

    # Initialize per-request token meter
    meter = reset_token_meter(cap=request_cap)

    # --- 1. BUDGET PRE-FLIGHT ---
    masked_init = mask_pii(question)[0]
    est_tokens = estimate_tokens(question)
    if config.MAX_INPUT_TOKENS is not None and est_tokens > config.MAX_INPUT_TOKENS:
        total_lat = round((time.perf_counter() - t_start) * 1000, 2)
        timings_ms["total"] = total_lat
        return PipelineResult(
            trace_id=trace_id,
            status="rejected",
            answer="Input exceeds maximum allowed token length.",
            intent="rejected",
            masked_question=masked_init,
            resolved_question=question,
            sources=[],
            ticket=None,
            grounded=False,
            top_similarity=None,
            approved=None,
            review_reason=None,
            cache_hit=False,
            guardrail_flags=guardrail_flags,
            tokens_used=0,
            cost_inr=0.0,
            latency_ms=total_lat,
            timings_ms=timings_ms,
        )

    # --- 2. INJECTION CHECK ---
    t0 = time.perf_counter()
    is_blocked, pattern_matched = detect_injection(question)
    timings_ms["guardrails_injection"] = round((time.perf_counter() - t0) * 1000, 2)

    if is_blocked:
        guardrail_flags.append(f"INJECTION_BLOCKED:{pattern_matched}")
        total_lat = round((time.perf_counter() - t_start) * 1000, 2)
        timings_ms["total"] = total_lat
        return PipelineResult(
            trace_id=trace_id,
            status="blocked",
            answer="Request blocked by prompt-injection guardrail.",
            intent="blocked",
            masked_question=masked_init,
            resolved_question=question,
            sources=[],
            ticket=None,
            grounded=False,
            top_similarity=None,
            approved=None,
            review_reason=None,
            cache_hit=False,
            guardrail_flags=guardrail_flags,
            latency_ms=total_lat,
            timings_ms=timings_ms,
        )

    # --- 3. PII MASKING ---
    t0 = time.perf_counter()
    masked_question, pii_count = mask_pii(question)
    timings_ms["guardrails_pii"] = round((time.perf_counter() - t0) * 1000, 2)
    if pii_count > 0:
        guardrail_flags.append(f"PII_MASKED:{pii_count}")

    # --- 4. MEMORY RESOLUTION ---
    t0 = time.perf_counter()
    mem_result = process_turn_memory(session_id, masked_question)
    timings_ms["memory_resolution"] = round((time.perf_counter() - t0) * 1000, 2)

    if mem_result.get("clarification"):
        clarification_answer = mem_result["clarification"]
        # Append turn to memory
        append_turn_to_history(session_id, masked_question, clarification_answer)
        total_lat = round((time.perf_counter() - t_start) * 1000, 2)
        timings_ms["total"] = total_lat
        tokens_used = meter.total_tokens
        return PipelineResult(
            trace_id=trace_id,
            status="ok",
            answer=clarification_answer,
            intent="fallback",
            masked_question=masked_question,
            resolved_question=masked_question,
            sources=[],
            ticket=None,
            grounded=True,
            top_similarity=None,
            approved=None,
            review_reason=None,
            cache_hit=False,
            guardrail_flags=guardrail_flags,
            tokens_used=tokens_used,
            cost_inr=compute_cost_inr(tokens_used),
            latency_ms=total_lat,
            timings_ms=timings_ms,
        )

    resolved_question = mem_result.get("resolved_question", masked_question)

    # --- 5. CACHE LOOKUP ---
    t0 = time.perf_counter()
    cached = cache.get(resolved_question)
    timings_ms["cache_lookup"] = round((time.perf_counter() - t0) * 1000, 2)

    if cached is not None:
        total_lat = round((time.perf_counter() - t_start) * 1000, 2)
        timings_ms["total"] = total_lat
        final_answer = cached["answer"]
        # Step 12: Memory append still happens on cache hit
        append_turn_to_history(session_id, masked_question, final_answer)
        tokens_used = meter.total_tokens
        return PipelineResult(
            trace_id=trace_id,
            status="ok",
            answer=final_answer,
            intent=cached["intent"],
            masked_question=masked_question,
            resolved_question=resolved_question,
            sources=cached["sources"],
            ticket=cached["ticket"],
            grounded=True,
            top_similarity=cached.get("top_similarity"),
            approved=cached.get("approved"),
            review_reason=cached.get("review_reason"),
            cache_hit=True,
            guardrail_flags=guardrail_flags,
            tokens_used=tokens_used,
            cost_inr=compute_cost_inr(tokens_used),
            latency_ms=total_lat,
            timings_ms=timings_ms,
        )

    # --- 6-10. CREW EXECUTION, VALIDATION, GROUNDEDNESS, REVIEW ---
    try:
        # --- 6 & 7. CREW EXECUTION (via asyncio.to_thread) ---
        t0 = time.perf_counter()
        crew_response: SupportResponse = await asyncio.to_thread(run_crew, resolved_question)
        timings_ms["crew_execution"] = round((time.perf_counter() - t0) * 1000, 2)

        # --- 8. SCHEMA VALIDATION ---
        t0 = time.perf_counter()
        SupportResponse.model_validate(crew_response)
        timings_ms["schema_validation"] = round((time.perf_counter() - t0) * 1000, 2)

        # --- 9. GROUNDEDNESS VERIFICATION ---
        t0 = time.perf_counter()
        ticket_data = None
        if crew_response.ticket:
            ticket_data = check_support_ticket_status(crew_response.ticket.record_id)

        rag_data = None
        if crew_response.intent in ["policy", "mixed"]:
            rag_query = resolved_question
            if crew_response.intent == "mixed" and ticket_data and ticket_data.get("found"):
                import re
                clean_q = re.sub(r"\(?ticket\s+OLA-\d{4}\)?", "", resolved_question, flags=re.IGNORECASE)
                clean_q = re.sub(r"OLA-\d{4}", "", clean_q, flags=re.IGNORECASE).strip()
                clean_q = re.sub(r"^[—\-\s]+|[—\-\s]+$", "", clean_q).strip()
                sev = ticket_data.get("severity", "")
                cat = ticket_data.get("category", "")
                days = ticket_data.get("days_since_created", "")
                rag_query = f"{clean_q} (for a {sev} {cat} ticket open {days} days)"
            rag_data = grounded_answer(rag_query, config.DEPLOYED_COLLECTION)

        is_grounded, ground_reason, unsupported_sents = check_grounded(
            crew_response, rag_result=rag_data, ticket_json=ticket_data
        )
        timings_ms["groundedness_check"] = round((time.perf_counter() - t0) * 1000, 2)

        top_sim = None
        if crew_response.intent in ["policy", "mixed"]:
            top_sim = rag_data.get("top_similarity") if rag_data else crew_response.top_similarity
        elif crew_response.top_similarity is not None and crew_response.intent not in ["ticket"]:
            top_sim = crew_response.top_similarity

        if not is_grounded:
            status = "refused"
            final_intent = "refused"
            if ground_reason == "TICKET_NOT_FOUND":
                final_answer = "No ticket with that ID exists."
            else:
                final_answer = config.FALLBACK_ANSWER
            final_sources = []
            final_ticket = None
        else:
            status = "ok"
            final_intent = crew_response.intent
            final_answer = crew_response.answer
            final_sources = crew_response.sources
            final_ticket = crew_response.ticket

        # --- 10. AUTOGEN REVIEW (Phase 7) ---
        approved = None
        review_reason = None
        if status == "ok":
            t0 = time.perf_counter()
            review_context = []
            if rag_data and rag_data.get("context"):
                review_context.extend(rag_data["context"])
            if ticket_data:
                import json
                review_context.append(json.dumps(ticket_data))

            verdict, _ = await run_review(final_answer, review_context)
            timings_ms["review"] = round((time.perf_counter() - t0) * 1000, 2)
            approved = verdict.approved
            review_reason = verdict.reason
            final_answer = verdict.final_answer

    except BudgetExceeded as e:
        total_lat = round((time.perf_counter() - t_start) * 1000, 2)
        timings_ms["total"] = total_lat
        tokens_at_abort = meter.used
        refused_info = f" on {e.refused_call}" if getattr(e, "refused_call", "") else ""
        return PipelineResult(
            trace_id=trace_id,
            status="rejected",
            answer=f"Request aborted: exceeded budget cap of {e.cap} tokens at stage '{e.stage}'{refused_info}.",
            intent="rejected",
            masked_question=masked_question,
            resolved_question=resolved_question,
            sources=[],
            ticket=None,
            grounded=False,
            top_similarity=None,
            approved=None,
            review_reason=None,
            cache_hit=False,
            guardrail_flags=guardrail_flags,
            tokens_used=tokens_at_abort,
            cost_inr=compute_cost_inr(tokens_at_abort),
            latency_ms=total_lat,
            timings_ms=timings_ms,
        )

    # --- 11. CACHE STORE ---
    if status == "ok":
        cache.put(
            resolved_question,
            {
                "answer": final_answer,
                "intent": final_intent,
                "sources": final_sources,
                "ticket": final_ticket,
                "top_similarity": top_sim,
                "approved": approved,
                "review_reason": review_reason,
                "status": status,
            },
        )

    # --- 12. MEMORY APPEND ---
    append_turn_to_history(session_id, masked_question, final_answer)

    total_lat = round((time.perf_counter() - t_start) * 1000, 2)
    timings_ms["total"] = total_lat
    tokens_used = meter.total_tokens
    return PipelineResult(
        trace_id=trace_id,
        status=status,
        answer=final_answer,
        intent=final_intent,
        masked_question=masked_question,
        resolved_question=resolved_question,
        sources=final_sources,
        ticket=final_ticket,
        grounded=is_grounded,
        top_similarity=top_sim,
        approved=approved,
        review_reason=review_reason,
        cache_hit=False,
        guardrail_flags=guardrail_flags,
        tokens_used=tokens_used,
        cost_inr=compute_cost_inr(tokens_used),
        latency_ms=total_lat,
        timings_ms=timings_ms,
    )


def answer_sync(
    session_id: str, question: str, request_cap: int | None = None
) -> PipelineResult:
    """Synchronous wrapper for answering queries."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(
                asyncio.run, answer(session_id, question, request_cap=request_cap)
            ).result()
    else:
        return asyncio.run(answer(session_id, question, request_cap=request_cap))
