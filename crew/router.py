"""Deterministic intent router for incoming queries (Section 12.1).

Rules:
- Contains OLA-\\d{4} AND policy keywords -> 'mixed'
- Contains OLA-\\d{4} only -> 'ticket'
- No ticket ID -> 'policy'
"""

import re
from typing import Literal

TICKET_PATTERN = re.compile(r"OLA-\d{4}", re.IGNORECASE)

POLICY_KEYWORDS = [
    "sla",
    "escalat",
    "refund",
    "credit",
    "priority",
    "policy",
    "handl",
    "who should",
    "who handles",
    "breach",
    "channel",
    "hour",
    "holiday",
    "repeat",
    "csat",
    "survey",
    "vip",
    "outage",
    "retention",
    "compensation",
    "response time",
    "resolution target",
]


def route_intent(query: str) -> Literal["policy", "ticket", "mixed"]:
    """Classify user query into policy, ticket, or mixed intent."""
    has_ticket = bool(TICKET_PATTERN.search(query))
    q_lower = query.lower()

    has_policy = any(kw in q_lower for kw in POLICY_KEYWORDS)

    if has_ticket and has_policy:
        return "mixed"
    elif has_ticket:
        return "ticket"
    else:
        return "policy"

