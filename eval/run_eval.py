"""Evaluation script running 15 evaluation queries across all topics (Section 16.4).

Produces:
1. transcripts/task13_eval.txt with 15-row table and 4 averages.
2. eval/results.json with full query, answer, context, scores, and rationales.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config  # Ground Rule 3: must be first import
import json
from utils.transcript import tee_transcript
from governance.cache import cache
from pipeline import answer_sync
from eval.testset import TEST_SET
from llm.mock_judge import build_judge_prompt, MockJudge, JudgeScores
from rag.retrieve import retrieve


def main():
    assert config.MOCK_LLM is True, "Evaluation must run with MOCK_LLM=True"

    with tee_transcript("task13_eval"):
        print("=" * 80)
        print("TASK 13: EVALUATION AT SCALE (15 QUERIES × 4 METRICS)")
        print("=" * 80)

        cache.clear()

        results = []
        scores_list: list[JudgeScores] = []

        first_prompt_printed = False

        for case in TEST_SET:
            session_id = f"eval_case_{case.id}"
            res = answer_sync(session_id, case.query)

            # Build context: EXACTLY the context the pipeline used (rag_context + ticket JSON)
            context_items = []
            if res.rag_context:
                context_items.extend(res.rag_context)
            elif res.sources:
                retrieved = retrieve(case.query, config.DEPLOYED_COLLECTION, k=config.TOP_K)
                for chunk in retrieved:
                    context_items.append(chunk["text"])
            if res.ticket:
                t_dict = res.ticket.model_dump()
                t_dict["found"] = True
                context_items.append(json.dumps(t_dict))

            prompt = build_judge_prompt(
                question=case.query,
                context=context_items,
                answer=res.answer,
                expected_facts=case.expected_facts,
                expected_behavior=case.expected_behavior,
            )

            if not first_prompt_printed:
                print("\n--- SAMPLE JUDGE PROMPT (CASE 1) ---")
                print(prompt)
                print("-" * 80)
                first_prompt_printed = True

            ticket_dict = res.ticket.model_dump() if res.ticket else None
            if ticket_dict:
                ticket_dict["found"] = True
            scores = MockJudge.judge(
                prompt,
                expected_doc=case.expected_doc,
                sources=res.sources,
                status=res.status,
                intent=res.intent,
                ticket=ticket_dict,
                top_similarity=res.top_similarity,
            )

            # Extra safety check for case 15
            if case.id == 15:
                log_file = config.LOGS_DIR / "requests.jsonl"
                if log_file.exists():
                    log_text = log_file.read_text(encoding="utf-8")
                    if "98765 43210" in log_text or "9876543210" in log_text:
                        scores.safety = 0.0

            scores_list.append(scores)
            results.append({
                "id": case.id,
                "topic": case.topic,
                "query": case.query,
                "expected_behavior": case.expected_behavior,
                "expected_doc": case.expected_doc,
                "expected_facts": case.expected_facts,
                "status": res.status,
                "intent": res.intent,
                "grounded": res.grounded,
                "answer": res.answer,
                "sources": res.sources,
                "top_similarity": res.top_similarity,
                "ticket": ticket_dict,
                "scores": scores.model_dump(),
            })

        # ---------------------------------------------------------------------
        # Print Results Table
        # ---------------------------------------------------------------------
        print("\n--- Evaluation Summary Table ---")
        header = f"{'ID':<3} | {'Topic':<22} | {'Status':<8} | {'Intent':<7} | {'Acc':<5} | {'Grd':<5} | {'Cmp':<5} | {'Saf':<5} | {'Rationale'}"
        print(header)
        print("-" * len(header))

        for r in results:
            sc = r["scores"]
            rat = sc["rationale"]
            short_rat = rat[:45] + "..." if len(rat) > 45 else rat
            print(
                f"{r['id']:<3} | "
                f"{r['topic']:<22} | "
                f"{r['status']:<8} | "
                f"{r['intent']:<7} | "
                f"{sc['accuracy']:<5.2f} | "
                f"{sc['grounding']:<5.2f} | "
                f"{sc['completeness']:<5.2f} | "
                f"{sc['safety']:<5.2f} | "
                f"{short_rat}"
            )

        print("-" * len(header))

        # ---------------------------------------------------------------------
        # Consistency Check
        # ---------------------------------------------------------------------
        warnings_found = []
        for r in results:
            if r["status"] == "ok" and r.get("grounded", False) and r["scores"]["grounding"] < 1.0:
                warn_msg = f"WARNING: Case {r['id']} [{r['topic']}] passed pipeline groundedness but judge grounding is {r['scores']['grounding']:.2f}"
                warnings_found.append(warn_msg)
                print(warn_msg)
        if not warnings_found:
            print("CONSISTENCY CHECK: All cases with status 'ok' and pipeline groundedness have judge grounding 1.0 ✓")

        # ---------------------------------------------------------------------
        # Print Detailed Case Rationales & Similarities
        # ---------------------------------------------------------------------
        print("\n--- Detailed Case Rationales & Retrieval Similarities ---")
        for r in results:
            sim_str = f" | top_sim: {r['top_similarity']:.4f}" if r['top_similarity'] is not None else ""
            print(f"Case {r['id']:<2} [{r['topic']}] (status: {r['status']}, intent: {r['intent']}{sim_str}):")
            print(f"  Rationale: {r['scores']['rationale']}")

        avg_acc = sum(s.accuracy for s in scores_list) / len(scores_list)
        avg_grd = sum(s.grounding for s in scores_list) / len(scores_list)
        avg_cmp = sum(s.completeness for s in scores_list) / len(scores_list)
        avg_saf = sum(s.safety for s in scores_list) / len(scores_list)

        perfect_count = sum(
            1 for s in scores_list
            if s.accuracy == 1.0 and s.grounding == 1.0 and s.completeness == 1.0 and s.safety == 1.0
        )

        print("\n--- Metric Macro-Averages (15 Cases) ---")
        print(f"Accuracy Average:     {avg_acc:.3f}")
        print(f"Grounding Average:    {avg_grd:.3f}")
        print(f"Completeness Average: {avg_cmp:.3f}")
        print(f"Safety Average:       {avg_saf:.3f}")
        print(f"Perfect 1.0/1.0/1.0/1.0 Count: {perfect_count} / {len(scores_list)} cases")

        # Save to eval/results.json
        results_file = ROOT_DIR / "eval" / "results.json"
        summary_payload = {
            "averages": {
                "accuracy": round(avg_acc, 3),
                "grounding": round(avg_grd, 3),
                "completeness": round(avg_cmp, 3),
                "safety": round(avg_saf, 3),
            },
            "perfect_cases": perfect_count,
            "total_cases": len(scores_list),
            "results": results,
        }
        results_file.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        print(f"\nDetailed evaluation results saved to {results_file} ✓")

        print("\n" + "=" * 80)
        print("TASK 13 EVALUATION COMPLETED SUCCESSFULLY ✓")
        print("=" * 80)


if __name__ == "__main__":
    main()
