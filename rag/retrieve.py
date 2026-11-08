"""Retrieval module with cosine similarity calculation (Section 9.5).

Ground Rule / Pitfall 5: Chroma returns distances in cosine space [0, 2],
so similarity = round(1 - distance, 4).
"""

import config  # Ground Rule 3: must be first import
from rag.embedder import embed_query
from rag.index import get_or_create_collection


def retrieve(query: str, collection_name: str, k: int = config.TOP_K) -> list[dict]:
    """Retrieve top-k chunks from collection_name using cosine similarity."""
    collection = get_or_create_collection(collection_name)
    query_emb = embed_query(query)

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    if not results or not results.get("ids") or not results["ids"][0]:
        return retrieved

    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for cid, text, meta, dist in zip(ids, docs, metas, distances):
        sim = round(1.0 - float(dist), 4)
        retrieved.append({
            "chunk_id": cid,
            "doc_id": meta.get("doc_id", ""),
            "title": meta.get("title", ""),
            "strategy": meta.get("strategy", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "text": text,
            "similarity": sim,
        })

    # Ensure results are sorted descending by similarity
    retrieved.sort(key=lambda x: x["similarity"], reverse=True)
    return retrieved

