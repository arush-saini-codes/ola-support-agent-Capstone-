"""Chunking evaluation module comparing fixed vs sentence strategies (Section 9.8).

Evaluates precision and recall across 5 in-scope queries with multi-document
relevance ground truth on both ola_kb_fixed and ola_kb_sentence collections.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config  # Ground Rule 3: must be first import after path setup
from rag.retrieve import retrieve
from utils.transcript import tee_transcript

EVAL_QUERIES = [
    {
        "id": "Q1",
        "query": "What is the resolution time for a P1 ticket?",
        "gold": {"02_sla_by_severity"},
    },
    {
        "id": "Q2",
        "query": "When do service credits expire and how much credit is given for a P2 SLA breach?",
        "gold": {"08_service_credits"},
    },
    {
        "id": "Q3",
        "query": "Who handles a ticket that has been open for more than 7 days?",
        "gold": {"03_escalation_matrix"},
    },
    {
        "id": "Q4",
        "query": "A P1 ticket breached its SLA — who handles it and what service credit applies?",
        "gold": {"02_sla_by_severity", "03_escalation_matrix", "08_service_credits"},
    },
    {
        "id": "Q5",
        "query": "A VIP rider complained twice about billing within a month — how is the ticket prioritised?",
        "gold": {"07_repeat_complaints", "10_vip_handling"},
    },
]


def evaluate_collection(collection_name: str) -> dict:
    """Evaluate retrieval precision and recall on a given collection."""
    print(f"\n{'=' * 80}")
    print(f"EVALUATION FOR COLLECTION: {collection_name}")
    print(f"{'=' * 80}")

    query_results = []
    for item in EVAL_QUERIES:
        qid = item["id"]
        qtext = item["query"]
        gold = item["gold"]

        top_chunks = retrieve(qtext, collection_name, k=config.TOP_K)
        retrieved_ids = [c["chunk_id"] for c in top_chunks]
        retrieved_docs = sorted(list(set(c["doc_id"] for c in top_chunks)))

        intersection = gold.intersection(set(retrieved_docs))
        p = len(intersection) / len(retrieved_docs) if retrieved_docs else 0.0
        r = len(intersection) / len(gold) if gold else 0.0

        print(f"\n[{qid}] \"{qtext}\"")
        print(f"  Retrieved Chunks : {retrieved_ids}")
        print(f"  Deduped Docs     : {retrieved_docs}")
        print(f"  Relevant Gold Set: {sorted(list(gold))}")
        print(f"  Intersection     : {sorted(list(intersection))}")
        print(f"  Arithmetic       : P = {len(intersection)}/{len(retrieved_docs)} = {p:.3f}, R = {len(intersection)}/{len(gold)} = {r:.3f}")

        query_results.append({
            "id": qid,
            "precision": p,
            "recall": r,
        })

    avg_p = sum(q["precision"] for q in query_results) / len(query_results)
    avg_r = sum(q["recall"] for q in query_results) / len(query_results)
    mean_score = (avg_p + avg_r) / 2.0

    print(f"\n--- MACRO-AVERAGES FOR {collection_name} ---")
    print(f"Macro-Average Precision: {avg_p:.3f}")
    print(f"Macro-Average Recall   : {avg_r:.3f}")
    print(f"Combined Mean (P+R)/2  : {mean_score:.3f}")

    return {
        "collection": collection_name,
        "query_results": query_results,
        "avg_precision": avg_p,
        "avg_recall": avg_r,
        "mean_score": mean_score,
    }


def run_chunking_eval():
    print("=" * 80)
    print("TASK 5: CHUNKING STRATEGY EVALUATION REPORT")
    print("=" * 80)

    fixed_res = evaluate_collection(config.COLLECTION_FIXED)
    sent_res = evaluate_collection(config.COLLECTION_SENTENCE)

    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE SUMMARY COMPARISON")
    print("=" * 80)
    header = f"{'Query':<8} | {'Fixed Prec':<12} | {'Fixed Rec':<12} | {'Sent Prec':<12} | {'Sent Rec':<12}"
    print(header)
    print("-" * len(header))

    for f_q, s_q in zip(fixed_res["query_results"], sent_res["query_results"]):
        print(f"{f_q['id']:<8} | {f_q['precision']:<12.3f} | {f_q['recall']:<12.3f} | {s_q['precision']:<12.3f} | {s_q['recall']:<12.3f}")

    print("-" * len(header))
    print(f"{'AVERAGE':<8} | {fixed_res['avg_precision']:<12.3f} | {fixed_res['avg_recall']:<12.3f} | {sent_res['avg_precision']:<12.3f} | {sent_res['avg_recall']:<12.3f}")

    # Determine recommendation based on higher (P+R)/2, ties -> sentence-based
    if sent_res["mean_score"] >= fixed_res["mean_score"]:
        winner = sent_res
        loser = fixed_res
        tie_note = " (including tie-breaker preference for natural sentence integrity)" if sent_res["mean_score"] == fixed_res["mean_score"] else ""
        reason = f"Sentence-based chunking preserves atomic policy facts cleanly without character-window boundary splits{tie_note}"
    else:
        winner = fixed_res
        loser = sent_res
        reason = "Fixed-size chunking achieved broader contextual span per chunk on this corpus"

    print("\n--- RECOMMENDATION & STRATEGY SELECTION ---")
    recommendation_text = (
        f"Based on empirical retrieval evaluation, collection '{winner['collection']}' achieved macro-average "
        f"precision of {winner['avg_precision']:.3f} and recall of {winner['avg_recall']:.3f} (mean score: {winner['mean_score']:.3f}), "
        f"compared to '{loser['collection']}' with precision of {loser['avg_precision']:.3f} and recall of {loser['avg_recall']:.3f} "
        f"(mean score: {loser['mean_score']:.3f}). {reason}. "
        f"Therefore, '{winner['collection']}' is recommended and selected as DEPLOYED_COLLECTION."
    )
    print(recommendation_text)

    return {
        "fixed": fixed_res,
        "sentence": sent_res,
        "winner": winner["collection"],
        "recommendation": recommendation_text,
    }


if __name__ == "__main__":
    with tee_transcript("task05_chunking_eval"):
        run_chunking_eval()

