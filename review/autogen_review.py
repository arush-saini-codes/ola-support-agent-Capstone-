"""AutoGen Review Stage for policy compliance and editing (Section 13).

Defines:
- ReviewVerdict: structured verdict returned by the final editor.
- build_review_team: creates the two-agent RoundRobinGroupChat team.
- run_review: asynchronously runs review on draft + context.
"""

import config  # Ground Rule 3: must be first import
import json
from typing import Any
from pydantic import BaseModel
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import StructuredMessage
from llm.mock_autogen_client import MockReviewClient


class ReviewVerdict(BaseModel):
    """Structured verdict emitted by the final editor."""
    approved: bool
    final_answer: str
    reason: str


def build_review_team() -> RoundRobinGroupChat:
    """Build the two-agent policy compliance and editing review team."""
    reviewer = AssistantAgent(
        "policy_compliance_reviewer",
        model_client=MockReviewClient(mode="reviewer"),
        system_message="Check every claim in the draft against the context...",
    )
    editor = AssistantAgent(
        "final_editor",
        model_client=MockReviewClient(mode="editor"),
        output_content_type=ReviewVerdict,
        system_message="Approve the draft unchanged or revise it...",
    )
    team = RoundRobinGroupChat(
        [reviewer, editor],
        max_turns=2,
        custom_message_types=[StructuredMessage[ReviewVerdict]],
    )
    return team


async def run_review(draft: str, context: list[str]) -> tuple[ReviewVerdict, list[Any]]:
    """Run the two-agent review on a draft and its retrieved context."""
    team = build_review_team()
    context_str = "\n".join(context)
    task_text = f"[[DRAFT]] {draft} [[/DRAFT]]\n[[CONTEXT]]\n{context_str}\n[[/CONTEXT]]"

    result = await team.run(task=task_text)

    # Extract verdict from the final message
    last_msg = result.messages[-1]
    raw_content = getattr(last_msg, "content", None)

    if isinstance(raw_content, ReviewVerdict):
        verdict = raw_content
    elif isinstance(raw_content, dict):
        verdict = ReviewVerdict.model_validate(raw_content)
    elif isinstance(raw_content, str):
        verdict = ReviewVerdict.model_validate_json(raw_content)
    else:
        raise ValueError(f"Unexpected review verdict format: {raw_content}")

    return verdict, result.messages

