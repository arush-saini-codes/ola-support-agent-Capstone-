"""Demo for Task 7: Multi-Agent CrewAI Crew Orchestration.

Demonstrates:
1. Policy query invoking rag_lookup only
2. Ticket query invoking check_support_ticket_status only
3. Mixed query invoking BOTH tools
4. Verification via TOOL_CALL_LOG and structured SupportResponse validation.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config  # Ground Rule 3: must be first import after path setup
import json
from crew.crew import run_crew
from llm.counters import TOOL_CALL_LOG, reset_counters, get_counter
from utils.transcript import tee_transcript

TEST_SCENARIOS = [
    (
        "A (Policy Query)",
        "What is the resolution time for a P1 ticket?",
        {"rag_lookup"},
    ),
    (
        "B (Ticket Query)",
        "What is the status of OLA-1001?",
        {"check_support_ticket_status"},
    ),
    (
        "C (Mixed Query)",
        "OLA-1027 is still open — who should it be escalated to?",
        {"rag_lookup", "check_support_ticket_status"},
    ),
]


def run_demo():
    print("=" * 80)
    print("TASK 7: MULTI-AGENT CREWAI ORCHESTRATION DEMO")
    print("=" * 80)

    all_passed = True

    for label, query, expected_tools in TEST_SCENARIOS:
        print(f"\n{'=' * 80}")
        print(f"RUNNING SCENARIO {label}: \"{query}\"")
        print(f"Expected Tools: {sorted(list(expected_tools))}")
        print("=" * 80)

        reset_counters()

        response = run_crew(query)

        called_tools = set(entry["tool"] for entry in TOOL_CALL_LOG)
        print("\n--- RESULTS & VERIFICATION ---")
        print(f"Detected Intent : {response.intent}")
        print(f"Tools Called    : {sorted(list(called_tools))}")
        print(f"Tool Calls Count: {get_counter('tool_calls')}")
        print(f"LLM Calls Count : {get_counter('crew_llm_calls')}")
        print("\nTOOL_CALL_LOG entries:")
        for log_entry in TOOL_CALL_LOG:
            print(f"  - {log_entry['tool']} with args: {log_entry['args']}")

        print(f"\nStructured SupportResponse JSON:\n{response.model_dump_json(indent=2)}")

        # Special detail for Scenario C
        if label.startswith("C"):
            rag_calls = [e for e in TOOL_CALL_LOG if e["tool"] == "rag_lookup"]
            if rag_calls:
                rewritten_q = rag_calls[0]["args"].get("query")
                print(f"\n[Scenario C Detail] Rewritten Query: \"{rewritten_q}\"")
                print(f"[Scenario C Detail] Top Similarity: {response.top_similarity} vs 0.41 (Grounded: {response.grounded})")
                print(f"[Scenario C Detail] Final Answer:\n{response.answer}")

        # Verification check
        tools_match = (called_tools == expected_tools)
        if not tools_match:
            print(f"[FAIL] Tool set mismatch! Expected {expected_tools}, got {called_tools}")
            all_passed = False
        else:
            print(f"[PASS] Exact expected tools invoked ✓")

    print("\n" + "=" * 80)
    print("TASK 7 OVERALL SUMMARY")
    print("=" * 80)
    print(f"Final Task 7 Status: {'ALL CHECKS PASSED ✓' if all_passed else 'DEMO FAILED ✗'}")
    return all_passed


if __name__ == "__main__":
    with tee_transcript("task07_crew"):
        success = run_demo()
    if not success:
        sys.exit(1)

