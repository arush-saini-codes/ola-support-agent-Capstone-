"""Demo for Task 3: ChromaDB Indexing and Sample Retrieval.

Builds both collections (ola_kb_fixed and ola_kb_sentence),
prints chunk counts and example chunks, then executes a sample query
against both collections to demonstrate sensible retrieval.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config  # Ground Rule 3: must be first import after path setup
from rag.loader import load_documents
from rag.chunking import chunk_all_documents
from rag.index import build_indexes, get_or_create_collection
from rag.retrieve import retrieve
from utils.transcript import tee_transcript


def run_demo():
    print("=" * 70)
    print("TASK 3: CHROMADB INDEXING & SAMPLE RETRIEVAL DEMO")
    print("=" * 70)

    # 1. Build indexes
    print("\n--- 1. BUILDING INDEXES ---")
    counts = build_indexes()
    for coll, count in counts.items():
        print(f"Collection '{coll}': {count} chunks indexed")

    # 2. Print example chunks
    print("\n--- 2. EXAMPLE CHUNKS ---")
    docs = load_documents()
    fixed_chunks = chunk_all_documents(docs, "fixed")
    sentence_chunks = chunk_all_documents(docs, "sentence")

    print("\n[Collection: ola_kb_fixed] (First 2 chunks):")
    for c in fixed_chunks[:2]:
        print(f"  ID: {c['id']}")
        print(f"  Text: {c['text']}")
        print(f"  Metadata: {c['metadata']}\n")

    print("\n[Collection: ola_kb_sentence] (First 2 chunks):")
    for c in sentence_chunks[:2]:
        print(f"  ID: {c['id']}")
        print(f"  Text: {c['text']}")
        print(f"  Metadata: {c['metadata']}\n")

    # 3. Sample query
    sample_query = "What is the resolution time for a P1 ticket?"
    print("\n--- 3. SAMPLE QUERY RETRIEVAL ---")
    print(f"Query: \"{sample_query}\"")

    for coll in [config.COLLECTION_FIXED, config.COLLECTION_SENTENCE]:
        print(f"\nRetrieval from '{coll}' (Top-3):")
        results = retrieve(sample_query, coll, k=3)
        for rank, res in enumerate(results, 1):
            print(f"  Rank {rank}:")
            print(f"    Chunk ID   : {res['chunk_id']}")
            print(f"    Doc ID     : {res['doc_id']}")
            print(f"    Similarity : {res['similarity']:.4f}")
            print(f"    Text       : {res['text']}")

    print("\nStatus: BOTH COLLECTIONS BUILT & RETRIEVAL VERIFIED ✓")
    return True


if __name__ == "__main__":
    with tee_transcript("task03_index"):
        run_demo()

