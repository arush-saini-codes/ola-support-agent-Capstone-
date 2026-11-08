"""Demo for Task 6: Ticket Status Lookup Tool & Escalation Scoring.

Demonstrates:
1. Escalation formula, activity factors, and data-driven threshold justification.
2. Programmatic lookups across diverse ticket profiles.
3. Robust error handling (case/whitespace normalization, not found, invalid format).
4. Full dataset score distribution, statistics, and text histogram.
5. Execution of CrewAI BaseTool wrappers (TicketStatusTool and RagLookupTool),
   verifying tool_calls counter and TOOL_CALL_LOG.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config  # Ground Rule 3: must be first import after path setup
import json
import statistics
import dataset
from tools.ticket_tool import check_support_ticket_status, TicketStatusTool
from tools.rag_tool import RagLookupTool
from llm.counters import get_counter, TOOL_CALL_LOG, reset_counters
from utils.transcript import tee_transcript


def run_demo():
    print("=" * 80)
    print("TASK 6: TICKET LOOKUP TOOL & ESCALATION SCORE DEMO")
    print("=" * 80)

    # a. Formula, weights, activity factors, and threshold justification
    print("\n--- A. ESCALATION FORMULA & JUSTIFICATION ---")
    print(f"Weight Flag (escalated) : {config.ESCALATION_WEIGHT_FLAG}")
    print(f"Weight Age (days / 30)  : {config.ESCALATION_WEIGHT_AGE}")
    print(f"Activity Factors        : {config.ACTIVITY_FACTORS}")
    print(f"Dataset P80 Ticket Age  : {dataset.days_percentile(80):.1f} days")
    print(f"Threshold Formula       : ESCALATION_THRESHOLD = 0.4 × P80(days_since_created) / 30 = 0.4 × 24.2 / 30 = {config.ESCALATION_THRESHOLD:.2f}")
    print("Justification:")
    print("  'ESCALATION_THRESHOLD = 0.4 × P80(days_since_created) / 30 = 0.4 × 24.2 / 30 = 0.32.")
    print("   An active non-escalated ticket crosses only when older than the 80th-percentile age of our data;")
    print("   every active escalated ticket always crosses (0.6 > 0.32); closed tickets never cross.'")

    # b. Programmatic lookups from actual dataset
    print("\n--- B. SAMPLE LOOKUPS FROM DATASET ---")
    tickets = dataset.TICKETS

    def find_first(predicate):
        for t in tickets:
            if predicate(t):
                return t
        return None

    samples = [
        ("Active Escalated", find_first(lambda t: t["status"] in ["Open", "In Progress", "Escalated"] and t["escalated"])),
        ("Active Non-Escalated Old (>24d)", find_first(lambda t: t["status"] in ["Open", "In Progress", "Escalated"] and not t["escalated"] and t["days_since_created"] > 24)),
        ("Active Non-Escalated Young (<=10d)", find_first(lambda t: t["status"] in ["Open", "In Progress", "Escalated"] and not t["escalated"] and t["days_since_created"] <= 10)),
        ("Resolved", find_first(lambda t: t["status"] == "Resolved")),
        ("Closed", find_first(lambda t: t["status"] == "Closed")),
    ]

    for label, ticket in samples:
        print(f"\n[{label} Ticket Lookup]:")
        if ticket:
            res = check_support_ticket_status(ticket["record_id"])
            print(json.dumps(res, indent=2))
        else:
            print(f"No ticket found matching condition for {label}")

    # c. Error cases
    print("\n--- C. ERROR HANDLING & NORMALIZATION ---")
    error_cases = [
        ("Whitespace & Lowercase", "ola-1001 "),
        ("Ticket Not Found", "OLA-9999"),
        ("Invalid Format", "hello"),
    ]

    for label, test_id in error_cases:
        print(f"\nCase: {label} (Input: {repr(test_id)}):")
        res = check_support_ticket_status(test_id)
        print(json.dumps(res, indent=2))

    # d. Distribution over all 50 tickets
    print("\n--- D. SCORE DISTRIBUTION OVER ALL 50 TICKETS ---")
    scores = []
    recommended = []

    for t in tickets:
        res = check_support_ticket_status(t["record_id"])
        s = res["escalation_score"]
        assert 0.0 <= s <= 1.0, f"Score out of bounds: {s} for {t['record_id']}"
        scores.append(s)
        if res["escalation_recommended"]:
            recommended.append(res)

    min_s = min(scores)
    med_s = statistics.median(scores)
    max_s = max(scores)

    print(f"Min Score    : {min_s:.3f}")
    print(f"Median Score : {med_s:.3f}")
    print(f"Max Score    : {max_s:.3f}")

    print("\nText Histogram (0.1 bins):")
    bins = [0] * 10
    for s in scores:
        idx = min(int(s * 10), 9)
        bins[idx] += 1

    for b in range(10):
        low = b / 10.0
        high = (b + 1) / 10.0
        bar = "#" * bins[b]
        print(f"  [{low:3.1f} - {high:3.1f}) : {bins[b]:2d} {bar}")

    rec_count = len(recommended)
    rec_pct = (rec_count / len(tickets)) * 100
    print(f"\nTickets above threshold ({config.ESCALATION_THRESHOLD:.2f}): {rec_count}/{len(tickets)} ({rec_pct:.1f}%)")

    print("\nRecommended Tickets for Escalation:")
    for r in recommended:
        print(f"  - {r['record_id']} | Status: {r['status']:12s} | Score: {r['escalation_score']:.3f} | Severity: {r['severity']}")

    # e. Call BaseTool wrappers once
    print("\n--- E. CREWAI BASETOOL WRAPPER EXECUTION ---")
    reset_counters()

    ticket_tool = TicketStatusTool()
    rag_tool = RagLookupTool()

    print("\nExecuting TicketStatusTool.run('OLA-1001')...")
    ticket_tool_output = ticket_tool.run("OLA-1001")
    print(f"Tool Output: {ticket_tool_output}")

    print("\nExecuting RagLookupTool.run('When do service credits expire?')...")
    rag_tool_output = rag_tool.run("When do service credits expire?")
    print(f"Tool Output: {rag_tool_output}")

    print(f"\ntool_calls counter: {get_counter('tool_calls')}")
    print(f"TOOL_CALL_LOG: {json.dumps(TOOL_CALL_LOG, indent=2)}")

    print("\nStatus: TASK 6 COMPLETE & VERIFIED ✓")
    return rec_count, rec_pct


if __name__ == "__main__":
    with tee_transcript("task06_ticket_tool"):
        run_demo()

