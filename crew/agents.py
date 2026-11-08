"""Agent factory and agent definitions for CrewAI crew (Section 12.2).

Creates:
1. Retrieval Agent (tools: [rag_lookup])
2. Lookup Agent (tools: [check_support_ticket_status])
3. Response Composer (tools: [])
"""

import config  # Ground Rule 3: must be first import
from crewai.agent.core import Agent
from crewai.tools import BaseTool
from llm.mock_crew_llm import MockCrewLLM
from tools.ticket_tool import TicketStatusTool
from tools.rag_tool import RagLookupTool
from governance.tool_policy import enforce_tool_policy


def build_agent(role: str, tools: list[BaseTool]) -> Agent:
    """Build and configure a CrewAI agent with dedicated MockCrewLLM."""
    enforce_tool_policy(role, tools)

    if role == "Retrieval Agent":
        goal = "Find relevant Ola policy guidelines and SLA terms from the knowledge base"
        backstory = "Expert policy researcher for Ola customer support operations"
        role_key = "retrieval"
    elif role == "Lookup Agent":
        goal = "Fetch accurate ticket records and compute escalation scores"
        backstory = "Support desk database specialist with access to customer ticket history"
        role_key = "lookup"
    elif role == "Response Composer":
        goal = "Synthesize gathered policy facts and ticket information into a clear final answer"
        backstory = "Senior customer support specialist ensuring grounded and compliant responses"
        role_key = "composer"
    else:
        goal = f"Perform duties of {role}"
        backstory = f"Specialist handling {role}"
        role_key = role.lower()

    llm = MockCrewLLM(model="mock-llm", role=role_key, tools=tools)

    agent = Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        tools=tools,
        allow_delegation=False,
        max_iter=3,
        verbose=True,
    )
    return agent


def create_crew_agents():
    """Create all three standardized crew agents."""
    rag_tool = RagLookupTool()
    ticket_tool = TicketStatusTool()

    retrieval_agent = build_agent("Retrieval Agent", [rag_tool])
    lookup_agent = build_agent("Lookup Agent", [ticket_tool])
    composer_agent = build_agent("Response Composer", [])

    return retrieval_agent, lookup_agent, composer_agent

