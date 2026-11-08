"""Ticket status lookup tool and escalation scoring (Section 10).

Implements:
1. check_support_ticket_status: look up tickets from dataset with designed
   escalation score formula (Section 10.2).
2. TicketStatusTool(BaseTool): CrewAI wrapper with TicketInput schema.
"""

import config  # Ground Rule 3: must be first import
import json
import re
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import dataset
from llm.counters import increment, log_tool_call

TICKET_ID_REGEX = re.compile(r"^OLA-\d{4}$")


def check_support_ticket_status(record_id: str) -> dict:
    """Look up an Ola support ticket and compute its escalation score.
    
    Normalizes input, validates format (^OLA-\\d{4}$), and returns ticket metadata
    along with the designed escalation score and full arithmetic explanation.
    """
    if not isinstance(record_id, str):
        return {
            "found": False,
            "record_id": str(record_id),
            "error": "INVALID_ID_FORMAT",
        }

    clean_id = record_id.strip().upper()
    if not TICKET_ID_REGEX.match(clean_id):
        return {
            "found": False,
            "record_id": record_id,
            "error": "INVALID_ID_FORMAT",
        }

    ticket = dataset.get_ticket(clean_id)
    if not ticket:
        return {
            "found": False,
            "record_id": clean_id,
            "error": "TICKET_NOT_FOUND",
        }

    category = ticket["category"]
    severity = ticket["severity"]
    status = ticket["status"]
    resolution_time_hours = ticket["resolution_time_hours"]
    days_since_created = ticket["days_since_created"]
    escalated = ticket["escalated"]

    # Designed Escalation Formula (Section 10.2)
    # age_signal normalized to [0, 1] over 30 days max dataset age
    age_signal = days_since_created / 30.0
    raw = (config.ESCALATION_WEIGHT_FLAG * float(escalated)) + (config.ESCALATION_WEIGHT_AGE * age_signal)
    activity_factor = config.ACTIVITY_FACTORS.get(status, 1.0)
    escalation_score = round(raw * activity_factor, 3)

    # Escalation recommended if score > threshold
    threshold = config.ESCALATION_THRESHOLD
    escalation_recommended = escalation_score > threshold

    rel_op = ">" if escalation_recommended else "<="
    score_explanation = (
        f"{config.ESCALATION_WEIGHT_FLAG}*{int(escalated)} + "
        f"{config.ESCALATION_WEIGHT_AGE}*({days_since_created}/30) = {raw:.3f}; "
        f"status {status} -> activity {activity_factor}; "
        f"final {escalation_score:.3f} {rel_op} threshold {threshold:.2f}"
    )

    return {
        "found": True,
        "record_id": clean_id,
        "category": category,
        "severity": severity,
        "status": status,
        "resolution_time_hours": resolution_time_hours,
        "days_since_created": days_since_created,
        "escalated": escalated,
        "escalation_score": escalation_score,
        "escalation_recommended": escalation_recommended,
        "score_explanation": score_explanation,
    }


class TicketInput(BaseModel):
    """Schema for ticket lookup tool input."""
    record_id: str = Field(..., description="The support ticket ID in OLA-XXXX format.")


class TicketStatusTool(BaseTool):
    """CrewAI tool for checking support ticket status and escalation score."""
    name: str = "check_support_ticket_status"
    description: str = (
        "Looks up the current status, resolution time, and escalation score "
        "of an Ola support ticket by its record ID."
    )
    args_schema: type[BaseModel] = TicketInput

    def _run(self, record_id: str) -> str:
        result = check_support_ticket_status(record_id)
        increment("tool_calls")
        log_tool_call(self.name, {"record_id": record_id})
        return json.dumps(result)

