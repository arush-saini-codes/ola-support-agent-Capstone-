"""Crew execution pipeline for running multi-agent workflows (Section 12.3).

Executes sequential crew consisting of:
- Retrieval Agent
- Lookup Agent
- Response Composer

Always constructed with all 3 agents and validated against SupportResponse.
"""

import config  # Ground Rule 3: must be first import
import json
import re
from crewai.crew import Crew
from crewai.process import Process
from crew.agents import create_crew_agents
from crew.tasks import build_tasks
from crew.schemas import SupportResponse
from governance.tool_policy import audit_crew, ToolPermissionError


def run_crew(question: str) -> SupportResponse:
    """Execute the sequential CrewAI crew on the user question and validate output."""
    retrieval_agent, lookup_agent, composer_agent = create_crew_agents()
    tasks, intent = build_tasks(question, retrieval_agent, lookup_agent, composer_agent)

    crew = Crew(
        agents=[retrieval_agent, lookup_agent, composer_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    audit_results = audit_crew(crew)
    for role, tools, is_allowed in audit_results:
        if not is_allowed:
            raise ToolPermissionError(f"Pre-kickoff audit failed: agent '{role}' has unauthorized tools {tools}")

    crew_output = crew.kickoff()

    # Extract validated SupportResponse
    if hasattr(crew_output, "pydantic") and isinstance(crew_output.pydantic, SupportResponse):
        return crew_output.pydantic

    raw_text = crew_output.raw if hasattr(crew_output, "raw") else str(crew_output)

    # Attempt direct validation
    try:
        return SupportResponse.model_validate_json(raw_text)
    except Exception:
        # If output was wrapped in markdown code blocks or thought prefixes, extract JSON
        m = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if m:
            return SupportResponse.model_validate_json(m.group(0))
        raise

