"""Singleton embedding model wrapper for Ola Support Agent (Section 9.3).

Uses sentence-transformers with 'all-MiniLM-L6-v2' (384-dimensional)
and normalize_embeddings=True. Loaded once per process.
"""

import config  # Ground Rule 3: must be first import
from sentence_transformers import SentenceTransformer

_EMBEDDER: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    """Return the singleton SentenceTransformer instance."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(config.EMBED_MODEL)
    return _EMBEDDER


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of text strings with normalize_embeddings=True."""
    if not texts:
        return []
    embedder = get_embedder()
    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]

