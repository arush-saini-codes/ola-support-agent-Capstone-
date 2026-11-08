"""Tool Policy & Least Autonomy Enforcement (Section 14.2).

Enforces:
- Static allow-list mapping tools to permitted agent roles.
- Constructor-time enforcement before any agent object is created.
- Pre-kickoff crew audit ensuring no unauthorized tool bindings exist.
"""

from typing import Any

TOOL_PERMISSIONS: dict[str, set[str]] = {
    "check_support_ticket_status": {"Lookup Agent"},
    "rag_lookup": {"Retrieval Agent"},
}


class ToolPermissionError(Exception):
    """Raised when an agent attempts to bind an unauthorized tool."""
    pass


def enforce_tool_policy(agent_role: str, tools: list[Any]) -> None:
    """Validate that every tool in tools is explicitly permitted for agent_role."""
    for tool in tools:
        tool_name = getattr(tool, "name", None) or str(tool)
        allowed_agents = TOOL_PERMISSIONS.get(tool_name, set())
        if agent_role not in allowed_agents:
            raise ToolPermissionError(
                f"Agent '{agent_role}' is not permitted to use tool '{tool_name}'. "
                f"Permitted agents: {sorted(list(allowed_agents)) or 'None'}"
            )


def audit_crew(crew: Any) -> list[tuple[str, list[str], bool]]:
    """Audit all agent-to-tool bindings in a crew.
    
    Returns:
        list[tuple[str, list[str], bool]]: [(role, tool_names, allowed), ...]
    """
    audit_results = []
    agents = getattr(crew, "agents", [])

    for agent in agents:
        role = getattr(agent, "role", "Unknown")
        tools = getattr(agent, "tools", []) or []
        tool_names = [getattr(t, "name", None) or str(t) for t in tools]

        is_allowed = True
        for t_name in tool_names:
            allowed_agents = TOOL_PERMISSIONS.get(t_name, set())
            if role not in allowed_agents:
                is_allowed = False
                break

        audit_results.append((role, tool_names, is_allowed))

    return audit_results

