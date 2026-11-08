"""Unit tests for ticket lookup tool and escalation scoring (Phase 4)."""

import config  # Ground Rule 3: must be first import
import pytest
import dataset
from tools.ticket_tool import check_support_ticket_status


def test_scores_within_bounds_for_all_tickets():
    """Verify that every ticket in the dataset scores between 0.0 and 1.0."""
    for t in dataset.TICKETS:
        res = check_support_ticket_status(t["record_id"])
        assert res["found"] is True
        score = res["escalation_score"]
        assert 0.0 <= score <= 1.0, f"Score {score} out of bounds for {t['record_id']}"


def test_closed_tickets_score_zero():
    """Verify that Closed tickets always receive an activity factor of 0.0 and score 0."""
    closed_tickets = [t for t in dataset.TICKETS if t["status"] == "Closed"]
    assert len(closed_tickets) > 0, "No Closed tickets found in dataset"

    for t in closed_tickets:
        res = check_support_ticket_status(t["record_id"])
        assert res["found"] is True
        assert res["escalation_score"] == 0.0
        assert res["escalation_recommended"] is False


def test_active_escalated_tickets_always_recommended():
    """Verify that active tickets with escalated=True always cross the 0.32 threshold."""
    active_escalated = [
        t for t in dataset.TICKETS
        if t["status"] in ["Open", "In Progress", "Escalated"] and t["escalated"]
    ]
    assert len(active_escalated) > 0, "No active escalated tickets found in dataset"

    for t in active_escalated:
        res = check_support_ticket_status(t["record_id"])
        assert res["found"] is True
        assert res["escalation_score"] >= 0.6  # 0.6 * 1.0 * 1.0 = 0.6 >= 0.32
        assert res["escalation_recommended"] is True


def test_invalid_and_unknown_ids():
    """Verify error handling for invalid formats and unknown ticket IDs."""
    r1 = check_support_ticket_status("hello")
    assert r1["found"] is False
    assert r1["error"] == "INVALID_ID_FORMAT"

    r2 = check_support_ticket_status("OLA-9999")
    assert r2["found"] is False
    assert r2["error"] == "TICKET_NOT_FOUND"

    r3 = check_support_ticket_status("")
    assert r3["found"] is False
    assert r3["error"] == "INVALID_ID_FORMAT"

    r4 = check_support_ticket_status(None)
    assert r4["found"] is False
    assert r4["error"] == "INVALID_ID_FORMAT"


def test_id_normalization():
    """Verify that lowercase and whitespace-padded IDs normalize correctly."""
    res = check_support_ticket_status("  ola-1001 \n")
    assert res["found"] is True
    assert res["record_id"] == "OLA-1001"
    assert res["status"] is not None

