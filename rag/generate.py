"""Grounded extractive generation module for Ola Support Agent (Section 9.7).

Under MOCK_LLM, generation is strictly extractive:
1. Retrieves top-k chunks from ChromaDB.
2. If top_similarity < threshold, returns FALLBACK_ANSWER with grounded=False.
3. Otherwise, splits retrieved chunks into individual sentences, embeds them,
   ranks them by cosine similarity to the query, and keeps the top 2 distinct
   sentences whose similarity >= threshold.
4. Formats answer with verbatim context sentences + '[Source: <doc_ids>]'.
5. Increments generation_calls counter.
"""

import config  # Ground Rule 3: must be first import
import re
import numpy as np
from rag.retrieve import retrieve
from rag.embedder import embed_texts, embed_query
from llm.counters import increment


def grounded_answer(query: str, collection_name: str | None = None) -> dict:
    """Generate an extractive grounded answer from the knowledge base."""
    target_coll = collection_name or config.DEPLOYED_COLLECTION or config.COLLECTION_SENTENCE
    threshold = config.SIMILARITY_THRESHOLDS.get(target_coll, config.SIMILARITY_THRESHOLD)

    # 1. Retrieve top-k chunks
    retrieved = retrieve(query, target_coll, k=config.TOP_K)
    context_texts = [r["text"] for r in retrieved]
    top_sim = retrieved[0]["similarity"] if retrieved else 0.0

    # 2. Check retrieval threshold
    if not retrieved or top_sim < threshold:
        return {
            "answer": config.FALLBACK_ANSWER,
            "sources": [],
            "top_similarity": top_sim,
            "grounded": False,
            "context": context_texts,
            "threshold_used": threshold,
        }

    # 3. Extractive sentence ranking
    candidate_sentences = []
    seen_texts = set()

    for item in retrieved:
        doc_id = item["doc_id"]
        # Split chunk on sentence boundaries
        raw_sents = re.split(r"(?<=[.!?])\s+", item["text"])
        for s in raw_sents:
            clean = s.strip()
            if clean and clean not in seen_texts:
                seen_texts.add(clean)
                candidate_sentences.append({"text": clean, "doc_id": doc_id})

    if not candidate_sentences:
        return {
            "answer": config.FALLBACK_ANSWER,
            "sources": [],
            "top_similarity": top_sim,
            "grounded": False,
            "context": context_texts,
            "threshold_used": threshold,
        }

    # Embed candidate sentences and compute similarities to query
    q_emb = np.array(embed_query(query))
    sent_embs = embed_texts([c["text"] for c in candidate_sentences])

    scored_sentences = []
    for item, s_emb in zip(candidate_sentences, sent_embs):
        sim = float(np.dot(q_emb, np.array(s_emb)))
        scored_sentences.append({
            "text": item["text"],
            "doc_id": item["doc_id"],
            "similarity": round(sim, 4),
        })

    # Rank descending by similarity
    scored_sentences.sort(key=lambda x: x["similarity"], reverse=True)

    # Keep top sentence if >= threshold; keep 2nd sentence ONLY if >= max(threshold, top_sentence_similarity - SECOND_SENTENCE_MARGIN)
    kept = []
    if scored_sentences and scored_sentences[0]["similarity"] >= threshold:
        top_sent_sim = scored_sentences[0]["similarity"]
        kept.append(scored_sentences[0])
        if len(scored_sentences) > 1:
            second_threshold = max(threshold, top_sent_sim - getattr(config, "SECOND_SENTENCE_MARGIN", 0.12))
            if scored_sentences[1]["similarity"] >= second_threshold:
                kept.append(scored_sentences[1])

    if not kept:
        return {
            "answer": config.FALLBACK_ANSWER,
            "sources": [],
            "top_similarity": top_sim,
            "grounded": False,
            "context": context_texts,
            "threshold_used": threshold,
        }

    # Format final answer
    body = " ".join(s["text"] for s in kept)
    sources = sorted(list(set(s["doc_id"] for s in kept)))
    final_answer = f"{body} [Source: {', '.join(sources)}]"

    # 4. Increment global generation calls counter
    increment("generation_calls")

    return {
        "answer": final_answer,
        "sources": sources,
        "top_similarity": top_sim,
        "grounded": True,
        "context": context_texts,
        "threshold_used": threshold,
    }

