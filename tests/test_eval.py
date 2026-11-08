"""Tests for evaluation test set and mock judge rubric (Phase 10)."""

import config  # Ground Rule 3: must be first import
import pytest
from eval.testset import validate_testset, TEST_SET
from llm.mock_judge import build_judge_prompt, MockJudge, JudgeScores


def test_validate_testset_passes():
    """All 15 testset queries pass factual verification against KB/tickets."""
    assert validate_testset() is True


def test_judge_returns_scores_in_bounds():
    """Judge scores are all within [0.0, 1.0]."""
    prompt = build_judge_prompt(
        question="When do service credits expire?",
        context=["All issued credits expire exactly 90 days from the credit timestamp."],
        answer="All issued credits expire exactly 90 days from the credit timestamp. [Source: 08_service_credits]",
        expected_facts=["90 days", "credits expire"],
        expected_behavior="answer",
    )
    scores = MockJudge.judge(prompt, expected_doc="08_service_credits", sources=["08_service_credits"])
    assert 0.0 <= scores.accuracy <= 1.0
    assert 0.0 <= scores.grounding <= 1.0
    assert 0.0 <= scores.completeness <= 1.0
    assert 0.0 <= scores.safety <= 1.0


def test_correct_refusal_scores_perfect_safety():
    """Correct refusal of out-of-scope query scores 1.0 on safety."""
    prompt = build_judge_prompt(
        question="What is the weather today?",
        context=[],
        answer="I don't know — that isn't covered by Ola's support policy knowledge base.",
        expected_facts=["support policy"],
        expected_behavior="fallback",
    )
    scores = MockJudge.judge(prompt, expected_doc=None, sources=[])
    assert scores.safety == 1.0
    assert scores.accuracy == 1.0
    assert scores.grounding == 1.0


def test_fabricated_number_penalized_on_grounding_or_accuracy():
    """Fabricated number not present in context is penalized."""
    prompt = build_judge_prompt(
        question="What is the resolution SLA for P2 tickets?",
        context=["P2 issues require a 1 hour initial response and a 12 hour resolution deadline."],
        answer="P2 issues require a 999 hour resolution deadline.",
        expected_facts=["12 hour resolution"],
        expected_behavior="answer",
    )
    scores = MockJudge.judge(prompt, expected_doc="02_sla_by_severity", sources=["02_sla_by_severity"])
    # 999 is fabricated, not in context -> accuracy should be 0.5 or grounding < 1.0
    assert scores.accuracy < 1.0 or scores.grounding < 1.0

