"""Task 14 Demo: AutoGen Multi-Agent Review Stage (Section 13).

Demonstrates:
1. Team configuration printing:
   - Agent names: policy_compliance_reviewer, final_editor
   - max_turns=2, custom_message_types=[StructuredMessage[ReviewVerdict]]
   - editor output_content_type=ReviewVerdict
2. Case 1 APPROVE (policy query "When do service credits expire?"):
   - Gets draft + context from crew execution
   - Runs AutoGen review
   - Prints all team messages, ReviewVerdict JSON, and asserts final_answer == draft -> UNCHANGED ✓
3. Case 2 REVISE (injected ungrounded sentence):
   - Appends "All P4 tickets also receive a Rs 500 service credit."
   - Reviewer emits UNSUPPORTED line, Editor removes it
   - Prints ReviewVerdict JSON, before/after, asserts approved=False and claim removed -> REVISED ✓
4. Case 3 APPROVE (mixed query "Check ticket OLA-1027 and tell me its resolution SLA"):
   - Context includes ticket JSON + RAG chunks
   - Runs AutoGen review -> approved=True
5. review_llm_calls counter accounting (+2 calls per review).
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config  # Ground Rule 3: must be first import
import json
import asyncio
from utils.transcript import tee_transcript
from llm.counters import get_counter
from review.autogen_review import build_review_team, run_review, ReviewVerdict
from crew.crew import run_crew
from rag.generate import grounded_answer
from tools.ticket_tool import check_support_ticket_status


async def async_main():
    with tee_transcript("task14_review"):
        print("=" * 80)
        print("TASK 14: AUTOGEN REVIEW STAGE DEMO")
        print("=" * 80)

        # ---------------------------------------------------------------------
        # 1. Print Team Configuration
        # ---------------------------------------------------------------------
        team = build_review_team()
        agent_names = [agent.name for agent in team._participants]
        print("\n--- AutoGen Team Configuration ---")
        print(f"Agent Names:               {agent_names}")
        print(f"Max Turns:                 2 (enforces exactly 1 turn per agent)")
        print(f"Custom Message Types:      [StructuredMessage[ReviewVerdict]]")
        print(f"Editor Output Content Type: ReviewVerdict")
        print("-" * 80)

        initial_calls = get_counter("review_llm_calls")
        print(f"Initial review_llm_calls: {initial_calls}")

        # ---------------------------------------------------------------------
        # Case 1: APPROVE (Policy Query: "When do service credits expire?")
        # ---------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("CASE 1: APPROVE POLICY DRAFT (ALL CLAIMS GROUNDED)")
        print("=" * 80)

        q1 = "When do service credits expire?"
        crew_resp_1 = await asyncio.to_thread(run_crew, q1)
        rag_data_1 = grounded_answer(q1, config.DEPLOYED_COLLECTION)
        context_1 = rag_data_1.get("context", [])

        draft_1 = crew_resp_1.answer
        print(f"Draft Answer:\n{draft_1}")
        print(f"Context Chunks ({len(context_1)}):")
        for c in context_1:
            print(f"  - {c}")

        calls_before = get_counter("review_llm_calls")
        verdict_1, messages_1 = await run_review(draft_1, context_1)
        calls_after = get_counter("review_llm_calls")

        print("\n--- Team Conversation Trace ---")
        for msg in messages_1:
            sender = getattr(msg, "source", "unknown")
            content = getattr(msg, "content", "")
            print(f"[{sender}]:\n{content}\n")

        print("--- Review Verdict JSON ---")
        print(verdict_1.model_dump_json(indent=2))

        assert verdict_1.approved is True, "Expected approved=True"
        assert verdict_1.final_answer == draft_1, "Expected final_answer == draft_1"
        assert calls_after - calls_before == 2, f"Expected 2 review calls, got {calls_after - calls_before}"
        print("\nResult: UNCHANGED ✓ (Approved unchanged byte-for-byte, +2 review_llm_calls)")

        # ---------------------------------------------------------------------
        # Case 2: REVISE (Ungrounded claim injected into draft)
        # ---------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("CASE 2: REVISE UNGROUNDED CLAIM (INJECTED FABRICATED STATEMENT)")
        print("=" * 80)

        injected_claim = "All P4 tickets also receive a Rs 500 service credit."
        draft_2 = f"{draft_1} {injected_claim}"
        print(f"Draft with Injected Claim:\n{draft_2}")

        calls_before = get_counter("review_llm_calls")
        verdict_2, messages_2 = await run_review(draft_2, context_1)
        calls_after = get_counter("review_llm_calls")

        print("\n--- Team Conversation Trace ---")
        for msg in messages_2:
            sender = getattr(msg, "source", "unknown")
            content = getattr(msg, "content", "")
            print(f"[{sender}]:\n{content}\n")

        print("--- Review Verdict JSON ---")
        print(verdict_2.model_dump_json(indent=2))

        print(f"\nBefore: {draft_2}")
        print(f"After:  {verdict_2.final_answer}")

        assert verdict_2.approved is False, "Expected approved=False"
        assert injected_claim not in verdict_2.final_answer, "Injected claim was not removed!"
        assert calls_after - calls_before == 2, f"Expected 2 review calls, got {calls_after - calls_before}"
        print("\nResult: REVISED ✓ (Rejected hallucination and pruned cleanly, +2 review_llm_calls)")

        # ---------------------------------------------------------------------
        # Case 3: APPROVE (Mixed Query with Ticket JSON in Context)
        # ---------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("CASE 3: APPROVE MIXED TICKET + POLICY QUERY")
        print("=" * 80)

        q3 = "Check ticket OLA-1027 and tell me its resolution SLA"
        crew_resp_3 = await asyncio.to_thread(run_crew, q3)
        ticket_data_3 = check_support_ticket_status("OLA-1027")
        rag_data_3 = grounded_answer("What is the resolution SLA for a P2 General Inquiry ticket?", config.DEPLOYED_COLLECTION)

        context_3 = list(rag_data_3.get("context", []))
        context_3.append(json.dumps(ticket_data_3))

        draft_3 = crew_resp_3.answer
        print(f"Mixed Draft Answer:\n{draft_3}")

        calls_before = get_counter("review_llm_calls")
        verdict_3, messages_3 = await run_review(draft_3, context_3)
        calls_after = get_counter("review_llm_calls")

        print("\n--- Review Verdict JSON ---")
        print(verdict_3.model_dump_json(indent=2))

        assert verdict_3.approved is True, "Expected approved=True"
        assert calls_after - calls_before == 2, f"Expected 2 review calls, got {calls_after - calls_before}"
        print("\nResult: UNCHANGED ✓ (Mixed ticket+policy verified against context, +2 review_llm_calls)")

        final_calls = get_counter("review_llm_calls")
        print("\n" + "=" * 80)
        print(f"TASK 14 REVIEW DEMO COMPLETED SUCCESSFULLY")
        print(f"Total review_llm_calls: {initial_calls} -> {final_calls} (total delta: +6 calls)")
        print("=" * 80)


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
