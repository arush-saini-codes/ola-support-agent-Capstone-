"""Unit tests for governance, budget caps, and caching (Sections 14.1-14.4)."""

import pytest
import config
from governance.tool_policy import enforce_tool_policy, ToolPermissionError, audit_crew
from governance.budget import estimate_tokens, BudgetExceeded
from governance.cache import normalize, ResponseCache
from tools.ticket_tool import TicketStatusTool
from tools.rag_tool import RagLookupTool
from crew.agents import build_agent, create_crew_agents
from crewai.crew import Crew
from pipeline import answer_sync
from llm.counters import get_counter


def test_tool_policy_allows_correct_bindings():
    """Test that permitted tools can be bound without error."""
    ticket_tool = TicketStatusTool()
    rag_tool = RagLookupTool()

    # Allowed: Lookup Agent with TicketStatusTool
    enforce_tool_policy("Lookup Agent", [ticket_tool])

    # Allowed: Retrieval Agent with RagLookupTool
    enforce_tool_policy("Retrieval Agent", [rag_tool])

    # Allowed: Response Composer with no tools
    enforce_tool_policy("Response Composer", [])


def test_tool_policy_blocks_unauthorized_bindings():
    """Test that unauthorized tool bindings raise ToolPermissionError."""
    ticket_tool = TicketStatusTool()
    rag_tool = RagLookupTool()

    # Blocked: Composer with ticket tool
    with pytest.raises(ToolPermissionError):
        enforce_tool_policy("Response Composer", [ticket_tool])

    # Blocked: Composer with rag tool
    with pytest.raises(ToolPermissionError):
        enforce_tool_policy("Response Composer", [rag_tool])

    # Blocked: Retrieval Agent with ticket tool
    with pytest.raises(ToolPermissionError):
        enforce_tool_policy("Retrieval Agent", [ticket_tool])

    # Blocked: Lookup Agent with rag tool
    with pytest.raises(ToolPermissionError):
        enforce_tool_policy("Lookup Agent", [rag_tool])


def test_build_agent_raises_on_invalid_tool():
    """Test constructor-time check in build_agent."""
    ticket_tool = TicketStatusTool()
    with pytest.raises(ToolPermissionError):
        build_agent("Response Composer", [ticket_tool])


def test_audit_crew():
    """Test that audit_crew correctly identifies allowed bindings."""
    retrieval, lookup, composer = create_crew_agents()
    crew = Crew(agents=[retrieval, lookup, composer])
    results = audit_crew(crew)
    assert len(results) == 3
    assert all(allowed for _, _, allowed in results)


def test_oversized_input_rejection():
    """Test pre-flight rejection of oversized input."""
    base = "Please give me details on the policy. "
    oversized = base * (config.MAX_INPUT_TOKENS * 4 // len(base) + 10)
    assert estimate_tokens(oversized) > config.MAX_INPUT_TOKENS

    res = answer_sync("gov_test_oversized", oversized)
    assert res.status == "rejected"
    assert "exceeds maximum allowed token length" in res.answer


def test_cache_hit_zero_llm_delta():
    """Test that a cache hit results in zero delta in LLM calls."""
    q = "When do service credits expire?"
    # First query populates cache
    res1 = answer_sync("gov_cache_test_1", q)
    assert res1.status == "ok"

    # Second query should hit cache
    crew_before = get_counter("crew_llm_calls")
    rev_before = get_counter("review_llm_calls")

    res2 = answer_sync("gov_cache_test_2", "  WHEN DO service CREDITS expire?  ")
    crew_after = get_counter("crew_llm_calls")
    rev_after = get_counter("review_llm_calls")

    assert res2.cache_hit is True
    assert crew_after == crew_before
    assert rev_after == rev_before
    assert res2.answer == res1.answer


def test_cache_normalization():
    """Test question normalization rules (lowercase, whitespace, punctuation)."""
    assert normalize("  What is the SLA?  ") == "what is the sla"
    assert normalize("What   is    the SLA???") == "what is the sla"
    assert normalize("WHAT IS THE SLA!.") == "what is the sla"
    assert normalize("Check ticket OLA-1027.") == "check ticket ola-1027"


def test_cache_lru_eviction():
    """Test LRU eviction when CACHE_MAX_ENTRIES is exceeded."""
    mini_cache = ResponseCache(max_entries=2)
    mini_cache.put("q1", {"status": "ok", "answer": "a1"})
    mini_cache.put("q2", {"status": "ok", "answer": "a2"})
    assert mini_cache.get("q1") is not None
    # Now q1 was accessed, so q2 is LRU
    mini_cache.put("q3", {"status": "ok", "answer": "a3"})
    assert mini_cache.get("q2") is None  # Evicted
    assert mini_cache.get("q1") is not None
    assert mini_cache.get("q3") is not None

