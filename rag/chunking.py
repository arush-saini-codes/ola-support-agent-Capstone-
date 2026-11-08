"""Chunking strategies for Ola policy knowledge base (Section 9.2).

Implements:
1. Fixed-size with overlap: 160 char windows with 40 char overlap,
   snapped to whitespace boundaries so words are never cut (Pitfall 16).
2. Sentence-based: splits on sentence boundaries ((?<=[.!?])\\s+).
"""

import config  # Ground Rule 3: must be first import
import re
from typing import Literal


def chunk_fixed_size(
    doc_id: str,
    title: str,
    text: str,
    chunk_size: int = config.FIXED_CHUNK_SIZE,
    overlap: int = config.FIXED_CHUNK_OVERLAP,
) -> list[dict]:
    """Fixed-size chunking with overlap and whitespace snapping.
    
    Chunk ID format: {doc_id}::fixed::{i}
    Metadata: {doc_id, title, strategy='fixed', chunk_index=i}
    """
    step = chunk_size - overlap  # default: 160 - 40 = 120
    chunks = []
    start = 0
    i = 0
    text_len = len(text)

    while start < text_len:
        raw_end = start + chunk_size
        if raw_end >= text_len:
            end = text_len
        else:
            # Snap raw_end to nearest whitespace so words are never cut
            left_ws = text.rfind(" ", start, raw_end)
            right_ws = text.find(" ", raw_end)

            if left_ws == -1 and right_ws == -1:
                end = raw_end
            elif left_ws == -1:
                end = right_ws
            elif right_ws == -1:
                end = left_ws
            else:
                if (raw_end - left_ws) <= (right_ws - raw_end):
                    end = left_ws
                else:
                    end = right_ws

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "id": f"{doc_id}::fixed::{i}",
                "text": chunk_text,
                "metadata": {
                    "doc_id": doc_id,
                    "title": title,
                    "strategy": "fixed",
                    "chunk_index": i,
                },
            })
            i += 1

        if end >= text_len:
            break

        # Calculate next window start and snap to word boundary
        next_raw_start = start + step
        if next_raw_start >= text_len:
            break

        if next_raw_start > 0 and text[next_raw_start - 1] != " " and text[next_raw_start] != " ":
            # In the middle of a word: advance past the word
            sp = text.find(" ", next_raw_start)
            if sp != -1:
                while sp < text_len and text[sp] == " ":
                    sp += 1
                next_start = sp
            else:
                next_start = next_raw_start
        else:
            next_start = next_raw_start
            while next_start < text_len and text[next_start] == " ":
                next_start += 1

        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks


def chunk_sentence_based(doc_id: str, title: str, text: str) -> list[dict]:
    """Sentence-based chunking on boundary regex (?<=[.!?])\\s+.
    
    Chunk ID format: {doc_id}::sent::{i}
    Metadata: {doc_id, title, strategy='sentence', chunk_index=i}
    """
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    i = 0
    for s in raw_sentences:
        clean = s.strip()
        if clean:
            chunks.append({
                "id": f"{doc_id}::sent::{i}",
                "text": clean,
                "metadata": {
                    "doc_id": doc_id,
                    "title": title,
                    "strategy": "sentence",
                    "chunk_index": i,
                },
            })
            i += 1
    return chunks


def chunk_document(doc: dict, strategy: Literal["fixed", "sentence"]) -> list[dict]:
    """Chunk a single document dictionary {doc_id, title, text}."""
    if strategy == "fixed":
        return chunk_fixed_size(doc["doc_id"], doc["title"], doc["text"])
    elif strategy == "sentence":
        return chunk_sentence_based(doc["doc_id"], doc["title"], doc["text"])
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")


def chunk_all_documents(docs: list[dict], strategy: Literal["fixed", "sentence"]) -> list[dict]:
    """Chunk all loaded documents using the given strategy."""
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, strategy))
    return all_chunks

