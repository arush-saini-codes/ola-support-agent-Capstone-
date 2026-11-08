"""Master pipeline execution script for Ola Support Agent (Phase 11).

Runs the entire project pipeline from scratch:
1. Enforces MOCK_LLM=true and telemetry disablement.
2. Deletes chroma_db/ and logs/requests.jsonl for clean initialization.
3. Executes each script in order as a fail-fast subprocess.
4. Validates transcript generation and prints a final summary table.
"""

import os
import sys
import shutil
import subprocess
import time
from pathlib import Path

# Setup environment before any imports
os.environ["MOCK_LLM"] = "true"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

ROOT_DIR = Path(__file__).resolve().parent

# (script_rel_path, primary_transcript_name)
PIPELINE_STEPS = [
    ("dataset.py", "task01_dataset.txt"),
    ("demos/check_kb_facts.py", "(verified in console)"),
    ("demos/task02_kb.py", "task02_kb.txt"),
    ("demos/task03_index.py", "task03_index.txt"),
    ("rag/calibrate.py", "task04_calibration.txt"),
    ("demos/task04_grounded_generation.py", "task04_grounded_generation.txt"),
    ("rag/eval_chunking.py", "task05_chunking_eval.txt"),
    ("demos/task06_ticket_tool.py", "task06_ticket_tool.txt"),
    ("demos/task07_crew.py", "task07_crew.txt"),
    ("demos/task08_memory.py", "task08_memory_multiturn.txt"),
    ("demos/task09_schema.py", "task09_schema.txt"),
    ("demos/task10_guardrails.py", "task10_guardrails.txt"),
    ("demos/task14_review.py", "task14_review.txt"),
    ("demos/task15_governance.py", "task15_governance.txt"),
    ("demos/task15_budget.py", "task15_budget.txt"),
    ("demos/task16_cache.py", "task16_cache.txt"),
    ("demos/task11_api.py", "task11_api.txt"),
    ("demos/task12_logging.py", "task12_logging.txt"),
    ("eval/run_eval.py", "task13_eval.txt"),
]


def main():
    print("=" * 80)
    print("OLA DOMAIN SUPPORT AGENT: MASTER REPRODUCTION RUN (run_all.py)")
    print("=" * 80)
    print(f"Python interpreter: {sys.executable}")
    print(f"Root directory:     {ROOT_DIR}")
    print("MOCK_LLM:           true (zero network access, zero API keys)")
    print("-" * 80)

    # 1. Clean ChromaDB and requests.jsonl
    chroma_dir = ROOT_DIR / "chroma_db"
    if chroma_dir.exists():
        print("Cleaning existing chroma_db/ ...")
        shutil.rmtree(chroma_dir, ignore_errors=True)

    requests_log = ROOT_DIR / "logs" / "requests.jsonl"
    if requests_log.exists():
        print("Cleaning existing logs/requests.jsonl ...")
        try:
            requests_log.unlink()
        except Exception:
            pass

    # Ensure transcripts and logs dir exist
    (ROOT_DIR / "transcripts").mkdir(exist_ok=True)
    (ROOT_DIR / "logs").mkdir(exist_ok=True)

    summary_rows = []
    total_start = time.time()

    # 2. Run steps in order
    sub_env = os.environ.copy()
    sub_env["PYTHONPATH"] = str(ROOT_DIR)

    for step_idx, (script_name, transcript_name) in enumerate(PIPELINE_STEPS, start=1):
        script_path = ROOT_DIR / script_name
        assert script_path.exists(), f"Script not found: {script_path}"

        print(f"\n[{step_idx:02d}/{len(PIPELINE_STEPS):02d}] Running: python {script_name} ...")
        step_start = time.time()

        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(ROOT_DIR),
            env=sub_env,
            capture_output=False,
        )

        step_elapsed = time.time() - step_start

        if proc.returncode != 0:
            print(f"❌ Step failed: {script_name} exited with code {proc.returncode}")
            summary_rows.append((script_name, transcript_name, f"FAIL ({proc.returncode})", f"{step_elapsed:.1f}s"))
            print("\nPIPELINE ABORTED DUE TO FAILURE.")
            sys.exit(proc.returncode)
        else:
            print(f"✓ Step completed in {step_elapsed:.1f}s")
            summary_rows.append((script_name, transcript_name, "OK", f"{step_elapsed:.1f}s"))

    total_elapsed = time.time() - total_start

    # 3. Print Final Summary Table
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Script':<35} | {'Transcript':<32} | {'Status':<6} | {'Time'}"
    print(header)
    print("-" * len(header))
    for script, trans, status, elapsed in summary_rows:
        print(f"{script:<35} | {trans:<32} | {status:<6} | {elapsed}")
    print("-" * len(header))
    print(f"Total Pipeline Runtime: {total_elapsed:.1f}s")
    print("ALL STEPS COMPLETED SUCCESSFULLY ✓")
    print("=" * 80)


if __name__ == "__main__":
    main()

