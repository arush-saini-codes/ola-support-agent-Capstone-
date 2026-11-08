"""Demo for Task 4: Grounded Extractive Generation.

Evaluates 5 in-scope queries and 1 out-of-scope query using the
sentence-based collection (ola_kb_sentence), demonstrating extractive
generation for grounded answers and fallback refusal for out-of-scope questions.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config  # Ground Rule 3: must be first import after path setup
from rag.retrieve import retrieve
from rag.generate import grounded_answer
from utils.transcript import tee_transcript

TEST_QUERIES = [
    ("Q1 (In-scope)", "What is the resolution time for a P1 ticket?", True),
    ("Q2 (In-scope)", "When do service credits expire and how much credit is given for a P2 SLA breach?", True),
    ("Q3 (In-scope)", "Who handles a ticket that has been open for more than 7 days?", True),
    ("Q4 (In-scope)", "A P1 ticket breached its SLA — who handles it and what service credit applies?", True),
    ("Q5 (In-scope)", "A VIP rider complained twice about billing within a month — how is the ticket prioritised?", True),
    ("OUT-OF-SCOPE", "What's the weather in Mumbai tomorrow?", False),
]


def run_demo():
    print("=" * 80)
    print("TASK 4: GROUNDED EXTRACTIVE GENERATION DEMO")
    print(f"Target Collection: {config.COLLECTION_SENTENCE}")
    threshold = config.SIMILARITY_THRESHOLDS[config.COLLECTION_SENTENCE]
    print(f"Calibrated Threshold: {threshold:.2f}")
    print("=" * 80)

    grounded_count = 0
    fallback_triggered = False

    for label, query, is_in_scope in TEST_QUERIES:
        print(f"\n[{label}] Query: \"{query}\"")

        # Top-3 retrieval inspection
        top_chunks = retrieve(query, config.COLLECTION_SENTENCE, k=config.TOP_K)
        print("Top-3 Retrieved Chunks:")
        for idx, c in enumerate(top_chunks, 1):
            print(f"  {idx}. [{c['doc_id']}] (sim={c['similarity']:.4f}) : {c['text']}")

        # Generation
        res = grounded_answer(query, config.COLLECTION_SENTENCE)
        print(f"Top Similarity vs Threshold : {res['top_similarity']:.4f} vs {res['threshold_used']:.2f}")
        print(f"Grounded Flag               : {res['grounded']}")
        print(f"Sources                     : {res['sources']}")
        print(f"Final Answer                : {res['answer']}")

        if is_in_scope and res["grounded"]:
            grounded_count += 1
        elif not is_in_scope and not res["grounded"]:
            fallback_triggered = True

    print("\n" + "=" * 80)
    print("DEMO VERIFICATION SUMMARY")
    print("=" * 80)
    c1 = (grounded_count == 5)
    c2 = fallback_triggered
    overall_pass = c1 and c2

    print(f"[{'PASS' if c1 else 'FAIL'}] 5 in-scope queries produced grounded answers: {grounded_count}/5")
    print(f"[{'PASS' if c2 else 'FAIL'}] Out-of-scope query triggered fallback refusal: {fallback_triggered}")
    print(f"\nFinal Task 4 Status: {'ALL CHECKS PASSED ✓' if overall_pass else 'DEMO FAILED ✗'}")

    return overall_pass


if __name__ == "__main__":
    with tee_transcript("task04_grounded_generation"):
        success = run_demo()
    if not success:
        sys.exit(1)

