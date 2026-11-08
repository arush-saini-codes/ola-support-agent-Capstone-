"""KB fact check verification script (PRE-STEP C).

Asserts that key domain facts from Section 8 are present in each of the
12 knowledge base markdown files in kb/*.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config  # Ground Rule 3: first import after path setup
from rag.loader import load_documents

# Key facts required per document from Section 8
REQUIRED_FACTS = {
    "01_ticket_priority": [
        "P1 Critical", "P2 High", "P3 Medium", "P4 Low",
        "rider safety", "payment debited with no ride", "full app outage",
        "Rs 500", "locked", "single feature bug", "refund under Rs 500"
    ],
    "02_sla_by_severity": [
        "15 min", "4 hour", "1 hour", "12 hour", "24 hour", "72 hour",
        "SLA breach", "escalation"
    ],
    "03_escalation_matrix": [
        "L1", "L2", "L3", "L4", "SLA breach", "second complaint",
        "7 days", "safety", "legal"
    ],
    "04_refund_compensation": [
        "Ola Money", "24 hour", "5 to 7", "10 min", "Rs 200", "L2"
    ],
    "05_communication_channels": [
        "in-app", "email", "phone", "WhatsApp", "OTP", "card"
    ],
    "06_business_hours_holidays": [
        "P1", "P2", "9 AM", "9 PM", "national holidays",
        "Republic Day", "Independence Day", "Gandhi Jayanti", "Diwali"
    ],
    "07_repeat_complaints": [
        "30", "repeat", "priority", "L2", "root-cause"
    ],
    "08_service_credits": [
        "150", "100", "50", "P1", "P2", "P3", "P4", "90 days", "non-transferable", "cash"
    ],
    "09_feedback_collection": [
        "CSAT", "1 to 5", "1 hour", "2", "48 hour", "7 days"
    ],
    "10_vip_handling": [
        "VIP", "corporate", "50 rides", "queue", "priority", "L2"
    ],
    "11_outage_communication": [
        "5%", "10 min", "15 min", "30 min", "48 hour", "bulk-resolved"
    ],
    "12_ticket_data_retention": [
        "3 years", "1 year", "masked", "30 days", "90 days"
    ]
}


def check_facts():
    """Verify all 12 documents against the required fact keywords."""
    print("=" * 70)
    print("PRE-STEP C: KNOWLEDGE BASE FACT CHECK REPORT")
    print("=" * 70)

    docs = {d["doc_id"]: d["text"] for d in load_documents()}
    all_passed = True
    fixes_needed = []

    for doc_id, facts in REQUIRED_FACTS.items():
        text = docs.get(doc_id, "")
        if not text:
            print(f"[{'FAIL':4s}] {doc_id:<28} : Document missing or empty")
            all_passed = False
            continue

        missing = []
        for fact in facts:
            if fact.lower() not in text.lower():
                missing.append(fact)

        if missing:
            all_passed = False
            fixes_needed.append((doc_id, missing))
            print(f"[{'FAIL':4s}] {doc_id:<28} : Missing facts: {missing}")
        else:
            print(f"[{'PASS':4s}] {doc_id:<28} : All {len(facts)} key facts present ✓")

    print("\n--- FACT CHECK SUMMARY ---")
    if all_passed:
        print("ALL 12 DOCUMENTS PASSED FACT VERIFICATION ✓")
    else:
        print(f"VALIDATION FAILED: {len(fixes_needed)} documents need fixes: {fixes_needed}")

    return all_passed


if __name__ == "__main__":
    passed = check_facts()
    if not passed:
        sys.exit(1)

