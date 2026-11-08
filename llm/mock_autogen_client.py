"""Deterministic Mock LLM client for AutoGen review team (Sections 11.3 & 13).

Avoids scripted replays by computing review verdicts dynamically:
- Reviewer: splits the draft into sentences, runs check_sentence_claims against context.
- Editor: parses reviewer findings, produces structured ReviewVerdict JSON approving or revising.
"""

import config  # Ground Rule 3: must be first import
import json
import math
import re
from typing import Any, AsyncGenerator, Sequence
from pydantic import BaseModel
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    RequestUsage,
    ModelInfo,
    LLMMessage,
)
from autogen_core import CancellationToken
from llm.counters import increment, get_token_meter
from guardrails.groundedness import check_sentence_claims


class MockReviewClient(ChatCompletionClient):
    """Custom AutoGen model client powering the two-agent review team."""

    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode  # "reviewer" | "editor"
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

    @property
    def model_info(self) -> ModelInfo:
        return {
            "vision": False,
            "function_calling": False,
            "json_output": True,
            "family": "unknown",
            "structured_output": True,
        }

    @property
    def capabilities(self) -> ModelInfo:
        return self.model_info

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = [],
        tool_choice: Any = "auto",
        json_output: bool | type[BaseModel] | None = None,
        extra_create_args: dict[str, Any] = {},
        cancellation_token: CancellationToken | None = None,
    ) -> CreateResult:
        """Compute review verdict deterministically from conversation messages."""
        # Stringify conversation
        full_text = ""
        for m in messages:
            content = getattr(m, "content", "")
            if isinstance(content, str):
                full_text += content + "\n"
            elif isinstance(content, list):
                for part in content:
                    full_text += str(part) + "\n"

        prompt_tokens = math.ceil(len(full_text) / 4)
        get_token_meter().check_before_call(prompt_tokens, call_name=f"MockReviewClient({self.mode})")

        # Extract draft and context from the task message
        m_draft = re.search(r"\[\[DRAFT\]\](.*?)\[\[/DRAFT\]\]", full_text, re.DOTALL)
        draft = m_draft.group(1).strip() if m_draft else ""

        m_context = re.search(r"\[\[CONTEXT\]\](.*?)\[\[/CONTEXT\]\]", full_text, re.DOTALL)
        context_str = m_context.group(1).strip() if m_context else ""

        # Parse context lines into chunk list and ticket JSON
        context_chunks: list[str] = []
        ticket_json: dict | None = None
        for line in context_str.splitlines():
            line_s = line.strip()
            if not line_s:
                continue
            if line_s.startswith("{") and line_s.endswith("}"):
                try:
                    d = json.loads(line_s)
                    if isinstance(d, dict) and "record_id" in d:
                        ticket_json = d
                        continue
                except Exception:
                    pass
            context_chunks.append(line_s)

        if self.mode == "reviewer":
            # Split draft into sentences (ignoring source tags)
            cleaned_draft = re.sub(r"\[Source:\s*[^\]]+\]", "", draft).strip()
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned_draft) if s.strip()]

            unsupported = check_sentence_claims(sentences, context_chunks, ticket_json=ticket_json)
            if not unsupported:
                content = "ALL CLAIMS SUPPORTED"
            else:
                content = "\n".join(f"UNSUPPORTED: {s}" for s in unsupported)

        else:  # editor
            # Read reviewer output from messages
            unsupported_found: list[str] = []
            for m in messages:
                c_text = str(getattr(m, "content", ""))
                for line in c_text.splitlines():
                    if line.startswith("UNSUPPORTED:"):
                        unsupp_sent = line.replace("UNSUPPORTED:", "").strip()
                        if unsupp_sent:
                            unsupported_found.append(unsupp_sent)

            if not unsupported_found:
                # Approve unchanged byte-for-byte
                verdict = {
                    "approved": True,
                    "final_answer": draft,
                    "reason": "All claims supported by retrieved context.",
                }
            else:
                # Remove unsupported sentences from draft, preserving [Source: ...] tag
                m_src = re.search(r"(\[Source:\s*[^\]]+\])", draft)
                source_tag = m_src.group(1) if m_src else ""

                draft_without_src = re.sub(r"\[Source:\s*[^\]]+\]", "", draft).strip()
                raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", draft_without_src) if s.strip()]

                cleaned_sentences = []
                for sent in raw_sentences:
                    if not any(u in sent for u in unsupported_found):
                        cleaned_sentences.append(sent)

                if cleaned_sentences:
                    revised_answer = " ".join(cleaned_sentences)
                    if source_tag:
                        revised_answer = f"{revised_answer} {source_tag}"
                else:
                    revised_answer = config.FALLBACK_ANSWER

                verdict = {
                    "approved": False,
                    "final_answer": revised_answer,
                    "reason": f"Removed unsupported claim(s): {'; '.join(unsupported_found)}",
                }

            content = json.dumps(verdict)

        comp_tokens = math.ceil(len(content) / 4)
        increment("review_llm_calls")
        get_token_meter().add(prompt=prompt_tokens, completion=comp_tokens)
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += comp_tokens

        usage = RequestUsage(prompt_tokens=prompt_tokens, completion_tokens=comp_tokens)
        return CreateResult(finish_reason="stop", content=content, usage=usage, cached=False)

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Any] = [],
        tool_choice: Any = "auto",
        json_output: bool | type[BaseModel] | None = None,
        extra_create_args: dict[str, Any] = {},
        cancellation_token: CancellationToken | None = None,
    ) -> AsyncGenerator[CreateResult, None]:
        res = await self.create(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            json_output=json_output,
            extra_create_args=extra_create_args,
            cancellation_token=cancellation_token,
        )
        yield res

    def actual_usage(self) -> RequestUsage:
        return RequestUsage(
            prompt_tokens=self._total_prompt_tokens,
            completion_tokens=self._total_completion_tokens,
        )

    def total_usage(self) -> RequestUsage:
        return self.actual_usage()

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = []) -> int:
        total_len = sum(len(str(getattr(m, "content", ""))) for m in messages)
        return math.ceil(total_len / 4)

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = []) -> int:
        return max(0, 8192 - self.count_tokens(messages, tools=tools))

    async def close(self) -> None:
        pass

