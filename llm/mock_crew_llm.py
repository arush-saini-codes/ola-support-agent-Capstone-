"""Deterministic Mock LLM layer for CrewAI agents (Section 11.2).

Avoids Pitfalls 1 and 2:
1. Searches for 'Observation:' ONLY in the generated region (messages after the
   initial system + task messages), ignoring the ReAct template in the system prompt.
2. Dispatches tool arguments strictly by inspecting tool.args_schema field names
   ('record_id' vs 'query') rather than matching tool names.
"""

import config  # Ground Rule 3: must be first import
import json
import math
import re
from typing import Any
from pydantic import Field
from crewai.llms.base_llm import BaseLLM
from llm.counters import increment, get_token_meter
from crew.router import route_intent
from crew.schemas import SupportResponse, TicketInfo


class MockCrewLLM(BaseLLM):
    """Custom CrewAI LLM implementation executing deterministic ReAct flows."""

    role: str = ""
    tools: list[Any] = Field(default_factory=list)
    model: str = "mock-llm"
    response_format: Any = None

    def supports_function_calling(self) -> bool:
        """Forces CrewAI to take the text ReAct path so actions are visible."""
        return False

    def supports_stop_words(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 8192

    def call(
        self,
        messages: list[dict] | str,
        tools: list[dict] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Execute deterministic mock reasoning and return action or final answer."""
        # Format / stringify messages for token counting and debug
        if isinstance(messages, str):
            prompt_str = messages
            msg_list = [{"role": "user", "content": messages}]
        else:
            prompt_str = "\n".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
            msg_list = messages

        # Per-request token accounting: check BEFORE executing or charging
        tokens_est = math.ceil(len(prompt_str) / 4)
        get_token_meter().check_before_call(tokens_est, call_name=f"MockCrewLLM({self.role})")

        # Optional debug dump
        if config.MOCK_LLM_DEBUG:
            debug_path = config.LOGS_DIR / "mock_llm_debug.txt"
            with open(debug_path, "a", encoding="utf-8") as f:
                f.write(f"\n--- MockCrewLLM Call [Role: {self.role}] ---\n")
                f.write(prompt_str + "\n")

        # Extract user question from [[QUERY]] ... [[/QUERY]] markers
        query = ""
        m_query = re.search(r"\[\[QUERY\]\](.*?)\[\[/QUERY\]\]", prompt_str, re.DOTALL)
        if m_query:
            query = m_query.group(1).strip()

        # Split messages: TEMPLATE REGION = messages[:2], GENERATED REGION = messages[2:]
        generated_region = msg_list[2:] if len(msg_list) > 2 else []

        # Find Observation in generated region
        observation_text = None
        for m in generated_region:
            content = str(m.get("content", ""))
            if "Observation:" in content:
                obs_part = content.split("Observation:")[-1].strip()
                # Stop before user prompt in message if attached
                if "\nAnalyze the tool result" in obs_part:
                    obs_part = obs_part.split("\nAnalyze the tool result")[0].strip()
                observation_text = obs_part

        # ReAct state machine
        if not observation_text:
            # First turn: execute tool or compose
            if self.tools:
                tool = self.tools[0]
                fields = tool.args_schema.model_fields.keys()
                if "record_id" in fields:
                    m_id = re.search(r"OLA-\d{4}", query, re.IGNORECASE)
                    if not m_id:
                        return (
                            "Thought: no ticket referenced\n"
                            "Final Answer: {\"found\": false, \"error\": \"NO_TICKET_REFERENCED\"}"
                        )
                    ticket_id = m_id.group(0).upper()
                    args = {"record_id": ticket_id}
                elif "query" in fields:
                    rewritten_query = query
                    if self.role == "retrieval" and route_intent(query) == "mixed":
                        # Look for ticket lookup JSON in context (prompt_str)
                        ticket_data = None
                        for match in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", prompt_str):
                            try:
                                d = json.loads(match.group(0))
                                if isinstance(d, dict) and d.get("found") and "record_id" in d:
                                    ticket_data = d
                                    break
                            except Exception:
                                pass

                        if ticket_data:
                            # Clean ID and parentheticals
                            clean_q = re.sub(r"\(?ticket\s+OLA-\d{4}\)?", "", query, flags=re.IGNORECASE)
                            clean_q = re.sub(r"OLA-\d{4}", "", clean_q, flags=re.IGNORECASE).strip()
                            clean_q = re.sub(r"^[—\-\s]+|[—\-\s]+$", "", clean_q).strip()
                            sev = ticket_data.get("severity", "")
                            cat = ticket_data.get("category", "")
                            days = ticket_data.get("days_since_created", "")
                            rewritten_query = f"{clean_q} (for a {sev} {cat} ticket open {days} days)"

                    args = {"query": rewritten_query}
                else:
                    args = {}

                react_output = (
                    f"Thought: I should use {tool.name}.\n"
                    f"Action: {tool.name}\n"
                    f"Action Input: {json.dumps(args)}"
                )
                print(f"\n[ReAct Trace - {self.role}]\n{react_output}")
                increment("crew_llm_calls")
                get_token_meter().add(prompt=tokens_est)
                return react_output
            else:
                # Composer agent: merge task context into SupportResponse
                comp = self._compose_json(prompt_str, query)
                react_output = f"Thought: I now know the final answer\nFinal Answer: {comp}"
                print(f"\n[ReAct Trace - {self.role}]\n{react_output}")
                increment("crew_llm_calls")
                get_token_meter().add(prompt=tokens_est)
                return react_output
        else:
            # Second turn: Observation is present, return clean observation as final answer
            react_output = (
                f"Observation: {observation_text}\n"
                f"Thought: I now know the final answer\n"
                f"Final Answer: {observation_text}"
            )
            print(f"\n[ReAct Trace - {self.role}]\n{react_output}")
            increment("crew_llm_calls")
            get_token_meter().add(prompt=tokens_est)
            return f"Thought: I now know the final answer\nFinal Answer: {observation_text}"

    def _compose_json(self, context_str: str, query: str) -> str:
        """Compose SupportResponse JSON from task context."""
        intent = route_intent(query)

        # Extract all JSON blocks from context
        rag_data = None
        ticket_data = None

        # Search for JSON blocks in context
        for match in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", context_str):
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    if "sources" in data and "answer" in data:
                        rag_data = data
                    elif "found" in data and "record_id" in data:
                        ticket_data = data
            except Exception:
                pass

        ticket_info = None
        ticket_text = ""
        if ticket_data and ticket_data.get("found"):
            ticket_info = TicketInfo(
                record_id=ticket_data["record_id"],
                category=ticket_data["category"],
                severity=ticket_data["severity"],
                status=ticket_data["status"],
                resolution_time_hours=ticket_data["resolution_time_hours"],
                days_since_created=ticket_data["days_since_created"],
                escalated=ticket_data["escalated"],
                escalation_score=ticket_data["escalation_score"],
                escalation_recommended=ticket_data["escalation_recommended"],
            )
            rec_str = "escalation recommended" if ticket_data["escalation_recommended"] else "escalation not recommended"
            ticket_text = (
                f"Ticket {ticket_data['record_id']} ({ticket_data['category']}, {ticket_data['severity']}) "
                f"is {ticket_data['status']}, created {ticket_data['days_since_created']} days ago, "
                f"resolution time so far {ticket_data['resolution_time_hours']} h. "
                f"Escalation score {ticket_data['escalation_score']} — {rec_str}."
            )
        elif ticket_data and not ticket_data.get("found"):
            ticket_text = f"Ticket {ticket_data.get('record_id', '')} could not be found."

        # Policy text
        policy_text = ""
        sources = []
        grounded = True
        top_sim = None
        if rag_data:
            policy_text = rag_data.get("answer", "")
            sources = rag_data.get("sources", [])
            grounded = rag_data.get("grounded", True)
            top_sim = rag_data.get("top_similarity")

        # Combine based on intent
        if intent == "ticket":
            final_answer = ticket_text or "No ticket details available."
            sources = []
            grounded = True
            top_sim = None
        elif intent == "policy":
            final_answer = policy_text or config.FALLBACK_ANSWER
        else:  # mixed
            if ticket_text and policy_text:
                final_answer = f"{ticket_text}\n\nPolicy: {policy_text}"
            else:
                final_answer = ticket_text or policy_text or config.FALLBACK_ANSWER

        response = SupportResponse(
            answer=final_answer,
            intent=intent,
            sources=sources,
            ticket=ticket_info,
            grounded=grounded,
            top_similarity=top_sim,
            guardrail_flags=[],
        )
        return response.model_dump_json()

