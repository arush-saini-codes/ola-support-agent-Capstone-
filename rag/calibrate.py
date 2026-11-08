"""Similarity threshold calibration module (Section 9.6).

Evaluates in-scope vs out-of-scope queries across both ChromaDB collections,
measures observed cosine similarity clusters, and calculates the empirical
threshold as the midpoint of the separation gap.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config  # Ground Rule 3: must be first import after path setup
from rag.retrieve import retrieve
from utils.transcript import tee_transcript

IN_SCOPE_QUERIES = [
    "What is the resolution time for a P1 ticket?",
    "When do service credits expire?",
    "Who handles a ticket that has been open for more than 7 days?",
    "How long are billing dispute tickets retained?",
    "When is a CSAT survey sent to the customer?",
]

OUT_OF_SCOPE_QUERIES = [
    "What's the weather in Mumbai tomorrow?",
    "Recommend a good biryani place in Bangalore",
    "Who won the cricket world cup in 2011?",
]


def calibrate_collection(collection_name: str) -> dict:
    """Run calibration queries on a collection and compute threshold gap."""
    print(f"\n{'=' * 75}")
    print(f"CALIBRATION FOR COLLECTION: {collection_name}")
    print(f"{'=' * 75}")

    header = f"{'Type':<14} | {'Top-1 Doc ID':<26} | {'Top-1 Sim':<10} | {'Query'}"
    print(header)
    print("-" * len(header))

    in_scope_sims = []
    for q in IN_SCOPE_QUERIES:
        res = retrieve(q, collection_name, k=1)
        sim = res[0]["similarity"] if res else 0.0
        doc = res[0]["doc_id"] if res else "NONE"
        in_scope_sims.append(sim)
        print(f"{'IN-SCOPE':<14} | {doc:<26} | {sim:<10.4f} | {q}")

    out_scope_sims = []
    for q in OUT_OF_SCOPE_QUERIES:
        res = retrieve(q, collection_name, k=1)
        sim = res[0]["similarity"] if res else 0.0
        doc = res[0]["doc_id"] if res else "NONE"
        out_scope_sims.append(sim)
        print(f"{'OUT-OF-SCOPE':<14} | {doc:<26} | {sim:<10.4f} | {q}")

    min_in_scope = min(in_scope_sims)
    max_out_scope = max(out_scope_sims)
    gap = min_in_scope - max_out_scope
    midpoint = round((min_in_scope + max_out_scope) / 2.0, 2)

    print(f"\n--- METRICS FOR {collection_name} ---")
    print(f"Min In-Scope Similarity    : {min_in_scope:.4f}")
    print(f"Max Out-of-Scope Similarity: {max_out_scope:.4f}")
    print(f"Separation Gap             : {gap:.4f}")
    print(f"Suggested Threshold (mid)  : {midpoint:.2f}")

    if max_out_scope >= min_in_scope:
        print(f"\n[WARNING] Overlap detected! Max out-of-scope ({max_out_scope:.4f}) >= Min in-scope ({min_in_scope:.4f})")
    else:
        print(f"[OK] Clean separation! Gap = {gap:.4f} > 0")

    return {
        "collection": collection_name,
        "min_in_scope": min_in_scope,
        "max_out_scope": max_out_scope,
        "gap": gap,
        "suggested_threshold": midpoint,
    }


def run_calibration():
    print("=" * 75)
    print("TASK 4 PART 1: SIMILARITY THRESHOLD CALIBRATION")
    print("=" * 75)

    results = {}
    for coll in [config.COLLECTION_FIXED, config.COLLECTION_SENTENCE]:
        results[coll] = calibrate_collection(coll)

    print("\n" + "=" * 75)
    print("SUMMARY COMPARISON")
    print("=" * 75)
    for coll, metrics in results.items():
        print(f"Collection {coll:<20}: Min In={metrics['min_in_scope']:.4f}, "
              f"Max Out={metrics['max_out_scope']:.4f}, Gap={metrics['gap']:.4f}, "
              f"Suggested Threshold={metrics['suggested_threshold']:.2f}")

    return results


if __name__ == "__main__":
    with tee_transcript("task04_calibration"):
        run_calibration()

