"""Demo for Task 2: Knowledge Base verification.

Loads every policy document in kb/*.md, parses its title and prose,
computes sentence counts on sentence boundaries, and asserts compliance.
"""

import sys
from pathlib import Path

# sys.path fix so demos can be executed directly as `python demos/task02_kb.py`
# or as module `python -m demos.task02_kb`
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config  # Ground Rule 3: must be first import after path setup
import re
from utils.transcript import tee_transcript


def inspect_knowledge_base():
    """Load and validate all documents in kb/*.md."""
    print("=" * 70)
    print("TASK 2: KNOWLEDGE BASE VERIFICATION REPORT")
    print("=" * 70)

    kb_files = sorted(config.KB_DIR.glob("*.md"))
    print(f"\nFound {len(kb_files)} policy documents in {config.KB_DIR}:\n")

    header = f"{'Doc ID':<28} | {'Sentence Count':<15} | {'Status':<10} | {'Title'}"
    print(header)
    print("-" * len(header))

    all_passed = True
    doc_results = []

    for fpath in kb_files:
        doc_id = fpath.stem
        raw_text = fpath.read_text(encoding="utf-8").strip()
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

        if not lines:
            print(f"{doc_id:<28} | {0:<15} | {'FAIL':<10} | (empty file)")
            all_passed = False
            continue

        title = lines[0].lstrip("#").strip()
        prose = " ".join(lines[1:]).strip()

        # Split on sentence boundaries: (?<=[.!?])\s+
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", prose) if s.strip()]
        s_count = len(sentences)

        # Requirement: each doc has 2–5 sentences (specifically 3–5)
        passed = (2 <= s_count <= 5)
        if not passed:
            all_passed = False

        status = "PASS" if passed else "FAIL"
        print(f"{doc_id:<28} | {s_count:<15} | {status:<10} | {title}")
        doc_results.append((doc_id, title, s_count, passed))

    print("\n--- SUMMARY OF CHECKS ---")
    doc_count_passed = len(kb_files) >= 12
    sentence_counts_passed = all(r[3] for r in doc_results)
    overall_passed = doc_count_passed and sentence_counts_passed

    print(f"[{'PASS' if doc_count_passed else 'FAIL'}] Document count >= 12: found {len(kb_files)}")
    print(f"[{'PASS' if sentence_counts_passed else 'FAIL'}] All documents have 2–5 sentences: {'All valid' if sentence_counts_passed else 'Violations found'}")
    print(f"\nFinal Task 2 Status: {'ALL CHECKS PASSED ✓' if overall_passed else 'VALIDATION FAILED ✗'}")

    return overall_passed


if __name__ == "__main__":
    with tee_transcript("task02_kb"):
        success = inspect_knowledge_base()
    if not success:
        sys.exit(1)

