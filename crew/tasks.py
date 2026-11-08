"""Task builders for CrewAI crew workflows (Section 12.3).

Builds task chains according to detected intent:
- policy : retrieve_task -> compose_task
- ticket : lookup_task -> compose_task
- mixed  : retrieve_task -> lookup_task -> compose_task
"""

import config  # Ground Rule 3: must be first import
from crewai.task import Task
from crew.router import route_intent
from crew.schemas import SupportResponse


def build_tasks(query: str, retrieval_agent, lookup_agent, composer_agent) -> tuple[list[Task], str]:
    """Construct task chain tailored to the query intent."""
    intent = route_intent(query)

    retrieve_task = Task(
        description=f"Search the Ola policy KB for: [[QUERY]]{query}[[/QUERY]]",
        expected_output="The JSON result of rag_lookup",
        agent=retrieval_agent,
    )

    lookup_task = Task(
        description=f"Look up the ticket referenced in: [[QUERY]]{query}[[/QUERY]]",
        expected_output="The JSON result of the ticket tool",
        agent=lookup_agent,
    )

    if intent == "policy":
        context_tasks = [retrieve_task]
        active_tasks = [retrieve_task]
    elif intent == "ticket":
        context_tasks = [lookup_task]
        active_tasks = [lookup_task]
    else:  # mixed: lookup -> retrieve -> compose (retrieve gets context=[lookup_task])
        retrieve_task.context = [lookup_task]
        context_tasks = [lookup_task, retrieve_task]
        active_tasks = [lookup_task, retrieve_task]

    compose_task = Task(
        description=f"Compose final support response for: [[QUERY]]{query}[[/QUERY]]",
        expected_output="A complete validated SupportResponse JSON",
        agent=composer_agent,
        context=context_tasks,
        output_pydantic=SupportResponse,
    )

    active_tasks.append(compose_task)
    return active_tasks, intent

