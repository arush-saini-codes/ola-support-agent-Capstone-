"""Task 15 Demo: Budget Caps and Token Governance (Section 14.3).

Demonstrates:
1. Printing calibration measurements and chosen caps:
   - Measured normal requests: policy=2727, ticket=2504, mixed=4586 tokens
   - Caps: MAX_INPUT_TOKENS = 256, MAX_REQUEST_TOKENS = 7000 (1.5 * 4586 = 6879 -> 7000)
2. Normal request: "When do service credits expire?" -> allowed; tokens_used vs cap and cost_inr.
3. Oversized request: 6,000-char input -> rejected at pre-flight with 0 LLM/tool calls.
4. In-flight abort: mixed query run with request_cap=300 -> aborted mid-run with status="rejected".
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config  # Ground Rule 3: must be first import
from utils.transcript import tee_transcript
from llm.counters import get_counter
from governance.budget import estimate_tokens
from pipeline import answer_sync


def main():
    with tee_transcript("task15_budget"):
        print("=" * 80)
        print("TASK 15: PER-REQUEST BUDGET CAPS & TOKEN GOVERNANCE DEMO")
        print("=" * 80)

        # ---------------------------------------------------------------------
        # 1. Calibration Measurements & Chosen Caps
        # ---------------------------------------------------------------------
        print("\n--- 1. Token Budget Calibration Summary ---")
        print("Measured token consumption across standard query types:")
        print("  - Policy query: 2,727 tokens")
        print("  - Ticket query: 2,504 tokens")
        print("  - Mixed query:  4,586 tokens (peak consumption)")
        print("\nChosen Governance Caps:")
        print(f"  - MAX_INPUT_TOKENS:   {config.MAX_INPUT_TOKENS} tokens (pre-flight guard)")
        print(f"  - MAX_REQUEST_TOKENS: {config.MAX_REQUEST_TOKENS} tokens (1.5 * 4,586 = 6,879 -> 7,000 in-flight guard)")
        print(f"  - Cost Rate:          Rs {config.COST_PER_1K_TOKENS_INR:.2f} per 1,000 tokens")
        print("-" * 80)

        # ---------------------------------------------------------------------
        # 2. Normal Request Under Budget
        # ---------------------------------------------------------------------
        print("\n--- 2. Normal Request Execution (Under Budget) ---")
        normal_q = "When do service credits expire?"
        print(f"Query: \"{normal_q}\"")

        res_normal = answer_sync("budget_normal", normal_q)
        print(f"Status:      {res_normal.status} ✓")
        print(f"Tokens Used: {res_normal.tokens_used} / {config.MAX_REQUEST_TOKENS} (Cap)")
        print(f"Cost INR:    Rs {res_normal.cost_inr:.4f}")
        print(f"Answer:      {res_normal.answer[:80]}...")
        assert res_normal.status == "ok"
        assert res_normal.tokens_used <= config.MAX_REQUEST_TOKENS

        # ---------------------------------------------------------------------
        # 3. Oversized Request (Pre-Flight Rejection)
        # ---------------------------------------------------------------------
        print("\n--- 3. Oversized Request (Pre-Flight Rejection) ---")
        base_sentence = "Please explain the detailed terms and conditions of ticket handling. "
        oversized_q = base_sentence * (6000 // len(base_sentence) + 1)
        est = estimate_tokens(oversized_q)
        print(f"Input Length:      {len(oversized_q)} characters")
        print(f"Estimated Tokens:  {est} tokens vs MAX_INPUT_TOKENS ({config.MAX_INPUT_TOKENS})")

        crew_before = get_counter("crew_llm_calls")
        review_before = get_counter("review_llm_calls")
        tool_before = get_counter("tool_calls")
        gen_before = get_counter("generation_calls")

        res_oversized = answer_sync("budget_oversized", oversized_q)

        crew_after = get_counter("crew_llm_calls")
        review_after = get_counter("review_llm_calls")
        tool_after = get_counter("tool_calls")
        gen_after = get_counter("generation_calls")

        print(f"Status:            {res_oversized.status} ✓")
        print(f"Pipeline Answer:   {res_oversized.answer}")
        print(f"Notional Cost:     Rs {res_oversized.cost_inr:.4f}")
        print("Counter Verification (Zero LLM / Tool / Generation Work):")
        print(f"  crew_llm_calls:   {crew_before} -> {crew_after} (delta: {crew_after - crew_before})")
        print(f"  review_llm_calls: {review_before} -> {review_after} (delta: {review_after - review_before})")
        print(f"  tool_calls:       {tool_before} -> {tool_after} (delta: {tool_after - tool_before})")
        print(f"  generation_calls: {gen_before} -> {gen_after} (delta: {gen_after - gen_before})")

        assert res_oversized.status == "rejected"
        assert crew_after - crew_before == 0
        assert review_after - review_before == 0
        assert tool_after - tool_before == 0
        assert gen_after - gen_before == 0

        # ---------------------------------------------------------------------
        # 4. In-Flight Abort (Cap = 3000 Tokens on Mixed Query)
        # ---------------------------------------------------------------------
        print("\n--- 4. In-Flight Budget Abort Demo (Cap = 3000 Tokens on Mixed Query) ---")
        mixed_q = "Check ticket OLA-1027 and tell me its resolution SLA"
        inflight_cap = 3000
        print(f"Query: \"{mixed_q}\" with in-flight request_cap = {inflight_cap} tokens")

        res_inflight = answer_sync("budget_inflight", mixed_q, request_cap=inflight_cap)
        print(f"Status:            {res_inflight.status} ✓")
        print(f"Cap:               {inflight_cap} tokens")
        print(f"Tokens at Abort:   {res_inflight.tokens_used} tokens (must be <= {inflight_cap})")
        print(f"Stage:             in_flight")
        print(f"Pipeline Answer:   {res_inflight.answer}")
        print(f"Notional Cost:     Rs {res_inflight.cost_inr:.4f}")

        # Extract refused call name from pipeline answer if present
        refused_call = "Unknown"
        if " on " in res_inflight.answer:
            refused_call = res_inflight.answer.split(" on ")[-1].rstrip(".")
        print(f"Refused Call:      {refused_call}")

        assert res_inflight.status == "rejected"
        assert res_inflight.tokens_used <= inflight_cap, f"Expected tokens_used <= {inflight_cap}, got {res_inflight.tokens_used}"
        assert f"exceeded budget cap of {inflight_cap} tokens at stage 'in_flight'" in res_inflight.answer

        print("\n" + "=" * 80)
        print("TASK 15 BUDGET DEMO COMPLETED SUCCESSFULLY ✓")
        print("=" * 80)


if __name__ == "__main__":
    main()

