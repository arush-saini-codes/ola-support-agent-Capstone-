"""ChromaDB indexing for Ola Support Agent (Section 9.4).

Maintains two ChromaDB collections:
1. ola_kb_fixed (fixed-size chunking)
2. ola_kb_sentence (sentence-based chunking)

Collections use cosine space ({"hnsw:space": "cosine"}) and NO default
embedding function (Pitfall 4). Telemetry is disabled via Settings.
"""

import config  # Ground Rule 3: must be first import
import chromadb
from chromadb.config import Settings
from rag.loader import load_documents
from rag.chunking import chunk_all_documents
from rag.embedder import embed_texts

_CLIENT = None


def get_chroma_client():
    """Singleton PersistentClient with telemetry disabled (PRE-STEP A)."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
    return _CLIENT


def get_or_create_collection(collection_name: str):
    """Retrieve or create a collection with cosine metric and NO default embedding function."""
    client = get_chroma_client()
    # Explicitly pass embedding_function=None so Chroma does not download ONNX model
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )
    return collection


def index_chunks(collection_name: str, chunks: list[dict]) -> int:
    """Embed and upsert chunks into the specified collection (idempotent)."""
    if not chunks:
        return 0

    collection = get_or_create_collection(collection_name)
    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    embeddings = embed_texts(documents)

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(ids)


def build_indexes() -> dict[str, int]:
    """Load KB documents, chunk under both strategies, and index into ChromaDB."""
    docs = load_documents()

    # Fixed-size strategy
    fixed_chunks = chunk_all_documents(docs, "fixed")
    fixed_count = index_chunks(config.COLLECTION_FIXED, fixed_chunks)

    # Sentence-based strategy
    sentence_chunks = chunk_all_documents(docs, "sentence")
    sent_count = index_chunks(config.COLLECTION_SENTENCE, sentence_chunks)

    return {
        config.COLLECTION_FIXED: fixed_count,
        config.COLLECTION_SENTENCE: sent_count,
    }
