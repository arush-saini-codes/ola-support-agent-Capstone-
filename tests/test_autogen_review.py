"""Unit tests for AutoGen Review stage (Section 13, Task 14)."""

import asyncio
from review.autogen_review import run_review, ReviewVerdict


def test_review_approve_grounded():
    """Test that fully grounded draft is approved unchanged."""
    async def _run():
        draft = "All issued credits expire exactly 90 days from the credit timestamp. [Source: 08_service_credits]"
        context = [
            "All issued credits expire exactly 90 days from the credit timestamp and are strictly non-transferable and not redeemable for cash withdrawals."
        ]
        verdict, messages = await run_review(draft, context)
        assert verdict.approved is True
        assert verdict.final_answer == draft
        assert "All claims supported" in verdict.reason
        assert len(messages) >= 2

    asyncio.run(_run())


def test_review_revise_unsupported_claim():
    """Test that injected unsupported claim is detected by reviewer and removed by editor."""
    async def _run():
        valid_part = "All issued credits expire exactly 90 days from the credit timestamp. [Source: 08_service_credits]"
        injected_part = "All P4 tickets also receive a Rs 500 service credit."
        draft = f"{valid_part} {injected_part}"
        context = [
            "All issued credits expire exactly 90 days from the credit timestamp and are strictly non-transferable and not redeemable for cash withdrawals."
        ]
        verdict, messages = await run_review(draft, context)
        assert verdict.approved is False
        assert injected_part not in verdict.final_answer
        assert valid_part in verdict.final_answer
        assert "Removed unsupported claim(s)" in verdict.reason

    asyncio.run(_run())
