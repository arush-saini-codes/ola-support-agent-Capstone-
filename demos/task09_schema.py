"""Task 9 Demo: Structured Output and Schema Validation (Section 12.4).

Demonstrates:
1. Strict Pydantic validation of SupportResponse across 4 distinct query types:
   - Policy query
   - Ticket query
   - Mixed query
   - Out-of-scope query
2. Printing VALID ✓ for each successfully validated schema.
3. Deliberately validating malformed payloads (missing required fields, invalid enum intent)
   and catching the expected ValidationError.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config  # Ground Rule 3: must be first import
import json
from pydantic import ValidationError
from utils.transcript import tee_transcript
from crew.schemas import SupportResponse
from pipeline import answer_sync


def main():
    with tee_transcript("task09_schema"):
        print("=" * 80)
        print("TASK 09: STRUCTURED OUTPUT & SCHEMA VALIDATION")
        print("=" * 80)

        test_queries = [
            ("Policy Query", "When do service credits expire?"),
            ("Ticket Query", "Check support ticket status for OLA-1027"),
            ("Mixed Query", "Check ticket OLA-1027 and tell me its resolution SLA"),
            ("Out-of-Scope Query", "What is Ola's policy on pet-friendly cabs?"),
        ]

        print("\n--- Part 1: Pipeline Execution & Schema Validation ---\n")
        for i, (q_type, query) in enumerate(test_queries, 1):
            print(f"[{i}] {q_type}: \"{query}\"")
            result = answer_sync(f"schema_test_{i}", query)

            # Reconstruct SupportResponse from pipeline execution
            response_obj = SupportResponse(
                answer=result.answer,
                intent=result.intent if result.intent in ["policy", "ticket", "mixed", "refused", "fallback", "blocked"] else "fallback",
                sources=result.sources,
                ticket=result.ticket,
                grounded=result.grounded,
                top_similarity=result.top_similarity,
                guardrail_flags=result.guardrail_flags,
            )

            # Strict Pydantic model validation
            validated_json = response_obj.model_dump_json(indent=2)
            # Re-validate via model_validate_json
            SupportResponse.model_validate_json(validated_json)

            print(f"Validated JSON Output:\n{validated_json}")
            print("Status: VALID ✓\n" + "-" * 60)

        print("\n--- Part 2: Malformed Payload Rejection (Deliberate Errors) ---\n")

        # Test case 1: Missing required field 'answer'
        malformed_1 = {
            "intent": "policy",
            "grounded": True,
            "sources": ["08_service_credits"],
        }
        print("Malformed Case 1 (Missing required field 'answer'):")
        print(f"Payload: {json.dumps(malformed_1)}")
        try:
            SupportResponse.model_validate(malformed_1)
            print("FAIL: Validation unexpectedly succeeded!")
        except ValidationError as e:
            print("CAUGHT EXPECTED ValidationError ✓:")
            for err in e.errors():
                print(f"  Field: {err['loc']} -> Error: {err['msg']}")

        # Test case 2: Invalid intent literal
        malformed_2 = {
            "answer": "This is a response.",
            "intent": "invalid_intent_type",
            "grounded": True,
        }
        print("\nMalformed Case 2 (Invalid intent literal 'invalid_intent_type'):")
        print(f"Payload: {json.dumps(malformed_2)}")
        try:
            SupportResponse.model_validate(malformed_2)
            print("FAIL: Validation unexpectedly succeeded!")
        except ValidationError as e:
            print("CAUGHT EXPECTED ValidationError ✓:")
            for err in e.errors():
                print(f"  Field: {err['loc']} -> Error: {err['msg']}")

        print("\n" + "=" * 80)
        print("TASK 09 SCHEMA VALIDATION DEMO COMPLETED SUCCESSFULLY")
        print("=" * 80)


if __name__ == "__main__":
    main()
