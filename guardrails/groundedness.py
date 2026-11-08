"""Output Groundedness Guardrail (Section 12.6).

Verifies that model responses are fully grounded in factual context:
1. Retrieval level: asserts top_similarity >= threshold when RAG is used.
2. Ticket level: verifies record_id, numbers, and category/severity/status match tool output.
3. Claim level: verifies sentences have >= 60% content word overlap and all numbers
   present in retrieved context.
"""

import config  # Ground Rule 3: must be first import
import re
from crew.schemas import SupportResponse

STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "and", "or", "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "this", "that", "these", "those", "can", "will", "as",
    "if", "than", "so", "no", "not", "also", "into", "all", "any", "each",
    "every", "more", "most", "other", "some", "such", "up", "out", "has",
    "have", "had", "do", "does", "did", "but", "about", "above", "after"
}


def check_grounded(
    response: SupportResponse,
    rag_result: dict | None = None,
    ticket_json: dict | None = None,
) -> tuple[bool, str, list[str]]:
    """Validate groundedness of a response against context and tool outputs.
    
    Returns:
        tuple[bool, str, list[str]]: (is_grounded, reason, unsupported_sentences)
    """
    # 1. Ticket existence check for ticket queries
    if response.intent in ["ticket", "mixed"] and ticket_json:
        if not ticket_json.get("found"):
            return False, "TICKET_NOT_FOUND", []

    # 2. Retrieval-level threshold check
    if response.intent in ["policy", "mixed"] and rag_result:
        threshold = rag_result.get("threshold_used", config.SIMILARITY_THRESHOLD)
        top_sim = rag_result.get("top_similarity") or 0.0
        if top_sim < threshold:
            return False, "RETRIEVAL_BELOW_THRESHOLD", []

def check_sentence_claims(
    sentences: list[str],
    context_chunks: list[str],
    ticket_json: dict | None = None,
) -> list[str]:
    """Verify that every sentence is grounded in either ticket data or retrieved context.
    
    Returns:
        list[str]: List of sentences deemed unsupported.
    """
    context_text = " ".join(context_chunks).lower()
    unsupported_sentences = []

    for sentence in sentences:
        if sentence.startswith("Ticket OLA-"):
            # Ticket claim validation
            if not ticket_json or not ticket_json.get("found"):
                unsupported_sentences.append(sentence)
                continue

            rec_id = ticket_json.get("record_id", "")
            if rec_id not in sentence:
                unsupported_sentences.append(sentence)
                continue

            # Verify category, severity, status
            cat = ticket_json.get("category", "")
            sev = ticket_json.get("severity", "")
            stat = ticket_json.get("status", "")

            if cat and cat.lower() not in sentence.lower():
                unsupported_sentences.append(sentence)
                continue
            if sev and sev.lower() not in sentence.lower():
                unsupported_sentences.append(sentence)
                continue
            if stat and stat.lower() not in sentence.lower():
                unsupported_sentences.append(sentence)
                continue

            # Check numbers against ticket JSON values
            valid_nums = {
                rec_id.split("-")[-1],  # e.g., "1027"
                str(ticket_json.get("resolution_time_hours")),
                str(ticket_json.get("days_since_created")),
                str(ticket_json.get("escalation_score")),
                f"{ticket_json.get('escalation_score', 0):.2f}",
                f"{ticket_json.get('escalation_score', 0):.3f}",
            }
            sent_nums = re.findall(r"\b\d+(?:\.\d+)?\b", sentence)
            for num in sent_nums:
                if num not in valid_nums:
                    unsupported_sentences.append(sentence)
                    break

        elif ticket_json and ticket_json.get("found") and "escalation score" in sentence.lower():
            # Escalation score claim validation
            valid_nums = {
                str(ticket_json.get("escalation_score")),
                f"{ticket_json.get('escalation_score', 0):.2f}",
                f"{ticket_json.get('escalation_score', 0):.3f}",
            }
            sent_nums = re.findall(r"\b\d+(?:\.\d+)?\b", sentence)
            for num in sent_nums:
                if num not in valid_nums:
                    unsupported_sentences.append(sentence)
                    break

        else:
            # Policy sentence validation against retrieved context
            if not context_text:
                # If no context exists, sentence is unsupported unless it's the standard fallback
                if sentence != config.FALLBACK_ANSWER:
                    unsupported_sentences.append(sentence)
                continue

            # Check numbers in policy sentence
            sent_nums = re.findall(r"\b\d+(?:\.\d+)?\b", sentence)
            nums_ok = all(num in context_text for num in sent_nums)
            if not nums_ok:
                unsupported_sentences.append(sentence)
                continue

            # Check content words (>= 60% overlap)
            words = re.findall(r"\b[a-zA-Z]{2,}\b", sentence.lower())
            content_words = [w for w in words if w not in STOP_WORDS]
            if content_words:
                matched = [w for w in content_words if w in context_text]
                overlap = len(matched) / len(content_words)
                if overlap < 0.60:
                    unsupported_sentences.append(sentence)
            else:
                # If no content words, ensure at least some match
                if not any(w in context_text for w in words):
                    unsupported_sentences.append(sentence)

    return unsupported_sentences


def check_grounded(
    response: SupportResponse,
    rag_result: dict | None = None,
    ticket_json: dict | None = None,
) -> tuple[bool, str, list[str]]:
    """Validate groundedness of a response against context and tool outputs.
    
    Returns:
        tuple[bool, str, list[str]]: (is_grounded, reason, unsupported_sentences)
    """
    # 1. Ticket existence check for ticket queries
    if response.intent in ["ticket", "mixed"] and ticket_json:
        if not ticket_json.get("found"):
            return False, "TICKET_NOT_FOUND", []

    # 2. Retrieval-level threshold check
    if response.intent in ["policy", "mixed"] and rag_result:
        threshold = rag_result.get("threshold_used", config.SIMILARITY_THRESHOLD)
        top_sim = rag_result.get("top_similarity") or 0.0
        if top_sim < threshold:
            return False, "RETRIEVAL_BELOW_THRESHOLD", []

    # 3. Claim-level sentence verification
    raw_answer = response.answer
    # Strip any source tags: [Source: ...]
    cleaned_answer = re.sub(r"\[Source:\s*[^\]]+\]", "", raw_answer).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned_answer) if s.strip()]

    # Collect retrieved context
    context_chunks = rag_result.get("context", []) if rag_result else []
    unsupported_sentences = check_sentence_claims(sentences, context_chunks, ticket_json=ticket_json)

    if unsupported_sentences:
        return False, "UNSUPPORTED_CLAIMS", unsupported_sentences

    return True, "GROUNDED", []
