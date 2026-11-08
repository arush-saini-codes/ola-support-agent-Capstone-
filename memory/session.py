"""Session memory with reference resolution (Section 12.5).

Uses LangChain's InMemoryChatMessageHistory and RunnableWithMessageHistory.
Resolves pronoun references ('it', 'its', 'this ticket', 'that ticket', 'same ticket')
to the most recent ticket ID found in session history.
"""

import config  # Ground Rule 3: must be first import
import re
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage

# Store of active in-memory chat histories per session_id
_STORE: dict[str, InMemoryChatMessageHistory] = {}

REF_PATTERN = re.compile(r"\b(it|its|this ticket|that ticket|same ticket)\b", re.IGNORECASE)
TICKET_PATTERN = re.compile(r"OLA-\d{4}", re.IGNORECASE)


def get_history(session_id: str) -> InMemoryChatMessageHistory:
    """Retrieve or create an InMemoryChatMessageHistory for a session."""
    if session_id not in _STORE:
        _STORE[session_id] = InMemoryChatMessageHistory()
    return _STORE[session_id]


def clear_all_sessions():
    """Reset the session memory store."""
    _STORE.clear()


def resolve_references_and_route(inputs: dict) -> dict:
    """Runnable turn function that inspects session history to resolve ticket references."""
    question = inputs["question"]
    history = inputs.get("history", [])

    resolved_question = question
    clarification = None

    # Check if reference resolution is required (has pronouns and no ticket ID)
    if REF_PATTERN.search(question) and not TICKET_PATTERN.search(question):
        found_id = None
        for msg in reversed(history):
            content = str(msg.content)
            m = TICKET_PATTERN.search(content)
            if m:
                found_id = m.group(0).upper()
                break

        if found_id:
            resolved_question = f"{question} (ticket {found_id})"
        else:
            clarification = "Which ticket do you mean? Please share the ticket ID (e.g. OLA-1027)."

    return {
        "question": question,
        "resolved_question": resolved_question,
        "clarification": clarification,
        "answer": clarification or "PROCEED_TO_CREW",
    }


# RunnableWithMessageHistory as mandated by Section 12.5 (deprecation warning visible)
memory_chain = RunnableWithMessageHistory(
    RunnableLambda(resolve_references_and_route),
    get_history,
    input_messages_key="question",
    history_messages_key="history",
    output_messages_key="answer",
)


def process_turn_memory(session_id: str, masked_question: str) -> dict:
    """Process a turn through memory chain to perform reference resolution."""
    return memory_chain.invoke(
        {"question": masked_question},
        config={"configurable": {"session_id": session_id}},
    )


def append_turn_to_history(session_id: str, masked_question: str, final_answer: str):
    """Append or update the completed turn in session history."""
    hist = get_history(session_id)
    if hist.messages and isinstance(hist.messages[-1], AIMessage):
        hist.messages[-1].content = final_answer
    else:
        hist.add_message(HumanMessage(content=masked_question))
        hist.add_message(AIMessage(content=final_answer))
