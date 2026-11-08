"""Self-check script for README.md integrity."""

from pathlib import Path
import re

readme_path = Path(__file__).resolve().parent.parent / "README.md"
content = readme_path.read_text(encoding="utf-8")

required_strings = [
    "uniform(0.3, 1.6)",
    "160",
    "0.6",
    "Resolved",
    "final_answer",
    "policy_compliance_reviewer",
    "MAX_INPUT_TOKENS = 256",
    "ola-support-agent.md)",
]

forbidden_strings = [
    "file:///",
    "W_{sev}",
    "revised_answer",
    "D:\\",
    "d:\\",
    "D:/",
    "d:/",
]

print("=" * 60)
print("README.MD INTEGRITY SELF-CHECK")
print("=" * 60)

all_passed = True

print("\n--- Checking Required Strings ---")
for req in required_strings:
    if req in content:
        print(f"  [PASS] Found required string: '{req}'")
    else:
        print(f"  [FAIL] Missing required string: '{req}'")
        all_passed = False

# Special check for P80 formula (either 0.4 * P80 or 0.4 × P80)
if "0.4 * P80" in content or "0.4 × P80" in content or "0.4 \\times \\frac{P80" in content:
    print("  [PASS] Found required P80 derivation formula")
else:
    print("  [FAIL] Missing required P80 derivation formula")
    all_passed = False

print("\n--- Checking Forbidden Strings ---")
for forb in forbidden_strings:
    if forb in content:
        print(f"  [FAIL] Found forbidden string: '{forb}'")
        all_passed = False
    else:
        print(f"  [PASS] Forbidden string absent: '{forb}'")

print("\n" + "=" * 60)
if all_passed:
    print("ALL README CHECKS PASSED SUCCESSFULLY ✓")
else:
    print("SOME CHECKS FAILED ✗")
print("=" * 60)

if not all_passed:
    exit(1)
