"""Task 8 Demo: Multi-turn session memory and reference resolution (Section 12.5).

Demonstrates:
1. Session S1 (multi-turn):
   - Turn 1: "Check OLA-1027" -> ticket details
   - Turn 2: "What is its resolution SLA?" -> resolves to OLA-1027, P2 -> 12 h
   - Turn 3: "Who handles it if it has been open more than 7 days?" -> resolves to OLA-1027 -> L3
   Outputs mirrored to transcripts/task08_memory_multiturn.txt.

2. Session S2 (fresh session):
   - Turn 1: "What is its resolution SLA?" -> history length 0 -> returns clarification:
     "Which ticket do you mean? Please share the ticket ID (e.g. OLA-1027)."
   Outputs mirrored to transcripts/task08_memory_fresh.txt.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config  # Ground Rule 3: must be first import
from utils.transcript import tee_transcript
from memory.session import get_history, clear_all_sessions
from pipeline import answer_sync


def run_session_s1():
    """Run multi-turn conversational session S1."""
    session_id = "S1"
    questions = [
        "Check OLA-1027",
        "What is its resolution SLA?",
        "Who handles it if it has been open more than 7 days?",
    ]

    print("=" * 80)
    print("TASK 08: MULTI-TURN SESSION MEMORY (SESSION S1)")
    print("=" * 80)

    for i, q in enumerate(questions, 1):
        hist_before = len(get_history(session_id).messages)
        print(f"\n--- Turn {i} ---")
        print(f"History length before turn: {hist_before} messages ({hist_before // 2} turns)")
        print(f"User Query: {q}")

        result = answer_sync(session_id, q)

        print(f"Resolved Query: {result.resolved_question}")
        print(f"Intent: {result.intent}")
        print(f"Status: {result.status}")
        print(f"Answer:\n{result.answer}")
        if result.sources:
            print(f"Sources: {result.sources}")

        hist_after = len(get_history(session_id).messages)
        print(f"History length after turn: {hist_after} messages ({hist_after // 2} turns)")

    print("\n" + "=" * 80)
    print("SESSION S1 COMPLETED SUCCESSFULLY")
    print("=" * 80)


def run_session_s2():
    """Run fresh session S2 with no prior context."""
    session_id = "S2"
    question = "What is its resolution SLA?"

    print("=" * 80)
    print("TASK 08: FRESH SESSION MEMORY (SESSION S2)")
    print("=" * 80)

    hist_before = len(get_history(session_id).messages)
    print(f"\n--- Turn 1 ---")
    print(f"History length before turn: {hist_before} messages ({hist_before // 2} turns)")
    print(f"User Query: {question}")

    result = answer_sync(session_id, question)

    print(f"Resolved Query: {result.resolved_question}")
    print(f"Intent: {result.intent}")
    print(f"Status: {result.status}")
    print(f"Answer:\n{result.answer}")

    hist_after = len(get_history(session_id).messages)
    print(f"History length after turn: {hist_after} messages")

    print("\n" + "=" * 80)
    print("SESSION S2 COMPLETED SUCCESSFULLY")
    print("=" * 80)


def main():
    clear_all_sessions()

    # Run 1: Multi-turn session S1 -> transcripts/task08_memory_multiturn.txt
    with tee_transcript("task08_memory_multiturn"):
        run_session_s1()

    # Run 2: Fresh session S2 -> transcripts/task08_memory_fresh.txt
    with tee_transcript("task08_memory_fresh"):
        run_session_s2()


if __name__ == "__main__":
    main()
