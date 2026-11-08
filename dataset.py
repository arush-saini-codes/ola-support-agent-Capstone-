"""Support ticket dataset generator for Ola Support Agent (Task 1).

Generates a deterministic synthetic dataset of 50 customer support tickets
using fixed random seeds and defined category/status/severity distributions.
"""

import config  # Ground Rule 3: must be first import
import json
import random
import sys
from collections import Counter
import numpy as np
from utils.transcript import tee_transcript

# --- Design Constants (Section 7.1) ---
SEED = 42
N_RECORDS = 50

CATEGORY_WEIGHTS = {
    "Billing": 0.28,
    "Technical Issue": 0.22,
    "Account Access": 0.18,
    "Product Defect": 0.14,
    "General Inquiry": 0.18,
}

STATUS_WEIGHTS = {
    "Open": 0.20,
    "In Progress": 0.22,
    "Escalated": 0.10,
    "Resolved": 0.28,
    "Closed": 0.20,
}

SEVERITY_WEIGHTS = {
    "P1": 0.10,
    "P2": 0.25,
    "P3": 0.40,
    "P4": 0.25,
}

SLA_HOURS = {
    "P1": 4,
    "P2": 12,
    "P3": 24,
    "P4": 72,
}

ESCALATED_BASE_PROB = 0.12


def generate_dataset(seed: int = SEED) -> list[dict]:
    """Generate deterministic support tickets according to Section 7.2.

    Fixed draw order per record:
    category -> status -> severity -> days_since_created -> escalated -> resolution_time_hours
    """
    rng = random.Random(seed)
    categories = list(CATEGORY_WEIGHTS.keys())
    cat_weights = list(CATEGORY_WEIGHTS.values())

    statuses = list(STATUS_WEIGHTS.keys())
    stat_weights = list(STATUS_WEIGHTS.values())

    severities = list(SEVERITY_WEIGHTS.keys())
    sev_weights = list(SEVERITY_WEIGHTS.values())

    records = []
    for i in range(1, N_RECORDS + 1):
        record_id = f"OLA-{1000 + i}"
        category = rng.choices(categories, weights=cat_weights, k=1)[0]
        status = rng.choices(statuses, weights=stat_weights, k=1)[0]
        severity = rng.choices(severities, weights=sev_weights, k=1)[0]
        days_since_created = rng.randint(0, 30)

        # The escalated rule is a generation rule, not hand-editing:
        # a ticket whose status is "Escalated" must logically carry the escalated flag.
        # Expected escalated share ≈ 10% + 12% × 90% ≈ 21%, inside the 10–30% band.
        if status == "Escalated":
            escalated = True
        else:
            escalated = rng.random() < ESCALATED_BASE_PROB

        # Range reasoning: resolution_time_hours ranges from 1 to 96 hours because
        # Ola's SLA tiers span 4 h (P1) to 72 h (P4), and the 1.6× upper multiplier
        # lets a realistic share of tickets breach their SLA.
        raw_res = SLA_HOURS[severity] * rng.uniform(0.3, 1.6)
        resolution_time_hours = round(min(96.0, max(1.0, raw_res)), 1)

        records.append({
            "record_id": record_id,
            "category": category,
            "status": status,
            "severity": severity,
            "days_since_created": days_since_created,
            "escalated": escalated,
            "resolution_time_hours": resolution_time_hours,
        })

    return records


# Public API: TICKETS built at import
TICKETS: list[dict] = generate_dataset(SEED)
_TICKET_INDEX: dict[str, dict] = {t["record_id"]: t for t in TICKETS}


def get_ticket(record_id: str) -> dict | None:
    """Public API: look up ticket by record_id (normalized: strip, uppercase)."""
    if not record_id:
        return None
    return _TICKET_INDEX.get(record_id.strip().upper())


def days_percentile(p: int) -> float:
    """Public API: compute the p-th percentile of days_since_created."""
    ages = [t["days_since_created"] for t in TICKETS]
    return float(np.percentile(ages, p))


def run_validation_and_report() -> bool:
    """Validate dataset against Section 7.4 rules and print full report."""
    print("=" * 60)
    print("TASK 1: DATASET GENERATION & VALIDATION REPORT")
    print("=" * 60)

    print("\n--- 1. DESIGN CONSTANTS ---")
    print(f"SEED = {SEED}")
    print(f"N_RECORDS = {N_RECORDS}")
    print(f"CATEGORY_WEIGHTS = {CATEGORY_WEIGHTS}")
    print(f"STATUS_WEIGHTS = {STATUS_WEIGHTS}")
    print(f"SEVERITY_WEIGHTS = {SEVERITY_WEIGHTS}")
    print(f"SLA_HOURS = {SLA_HOURS}")
    print(f"ESCALATED_BASE_PROB = {ESCALATED_BASE_PROB}")

    cat_counts = Counter(t["category"] for t in TICKETS)
    stat_counts = Counter(t["status"] for t in TICKETS)
    sev_counts = Counter(t["severity"] for t in TICKETS)

    print("\n--- 2. DISTRIBUTION BREAKDOWN ---")
    print("Category breakdown:")
    for cat in CATEGORY_WEIGHTS:
        cnt = cat_counts[cat]
        print(f"  - {cat:20s}: {cnt:2d} ({cnt / N_RECORDS * 100:.1f}%)")

    print("\nStatus breakdown:")
    for stat in STATUS_WEIGHTS:
        cnt = stat_counts[stat]
        print(f"  - {stat:20s}: {cnt:2d} ({cnt / N_RECORDS * 100:.1f}%)")

    print("\nSeverity breakdown:")
    for sev in SEVERITY_WEIGHTS:
        cnt = sev_counts[sev]
        print(f"  - {sev:20s}: {cnt:2d} ({cnt / N_RECORDS * 100:.1f}%)")

    escalated_count = sum(1 for t in TICKETS if t["escalated"])
    escalated_pct = (escalated_count / N_RECORDS) * 100
    print("\n--- 3. ESCALATION STATS ---")
    print(f"Total escalated tickets: {escalated_count}/{N_RECORDS} ({escalated_pct:.1f}%)")

    p50 = days_percentile(50)
    p80 = days_percentile(80)
    print("\n--- 4. AGE PERCENTILES (days_since_created) ---")
    print(f"P50 (Median age): {p50:.1f} days")
    print(f"P80 (80th percentile): {p80:.1f} days")

    # Section 7.4 Validation Checks
    checks = []
    c1 = len(TICKETS) >= 40
    checks.append(("N >= 40 records", c1, f"{len(TICKETS)} records"))

    c2 = all(cat_counts[c] >= 3 for c in CATEGORY_WEIGHTS)
    checks.append(("Each category >= 3 records", c2, f"Min category count = {min(cat_counts.values())}"))

    c3 = all(stat_counts[s] >= 1 for s in STATUS_WEIGHTS)
    checks.append(("Each status >= 1 record", c3, f"Min status count = {min(stat_counts.values())}"))

    c4 = 10.0 <= escalated_pct <= 30.0
    checks.append(("Escalated % in [10, 30]", c4, f"{escalated_pct:.1f}%"))

    print("\n--- 5. VALIDATION BLOCK ---")
    all_passed = True
    for desc, passed, detail in checks:
        status_str = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status_str}] {desc:32s} (Details: {detail})")

    # Write data/tickets.json
    out_path = config.DATA_DIR / "tickets.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(TICKETS, f, indent=2)
    print(f"\nWritten {len(TICKETS)} tickets to {out_path}")

    print(f"\nOverall Validation Status: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


if __name__ == "__main__":
    with tee_transcript("task01_dataset"):
        success = run_validation_and_report()
    if not success:
        sys.exit(1)
