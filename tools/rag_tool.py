"""RAG lookup tool wrapper for CrewAI (Section 10.5).

Deliberately named 'rag_lookup' (containing 'lookup') to verify that the
mock LLM layer dispatches arguments based on args_schema rather than tool names
(Pitfall 2).
"""

import config  # Ground Rule 3: must be first import
import json
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from rag.generate import grounded_answer
from llm.counters import increment, log_tool_call


class RagInput(BaseModel):
    """Schema for RAG knowledge base lookup query."""
    query: str = Field(..., description="The policy query string to look up in Ola's support knowledge base.")


class RagLookupTool(BaseTool):
    """CrewAI tool for retrieving grounded policy answers from the knowledge base."""
    name: str = "rag_lookup"
    description: str = "Searches the Ola support policy knowledge base for relevant guidelines, SLAs, and protocols."
    args_schema: type[BaseModel] = RagInput

    def _run(self, query: str) -> str:
        result = grounded_answer(query, config.DEPLOYED_COLLECTION)
        increment("tool_calls")
        log_tool_call(self.name, {"query": query})
        return json.dumps(result)

