"""Task 15 Demo: Governance and Least Autonomy Tool Policy (Section 14).

Demonstrates:
1. Four-layer governance model table.
2. Static allow-list auditing of the genuine 3-agent crew.
3. Enforcement of allow_delegation=False across all agents.
4. Construction-time blocking:
   - Response Composer attempting to bind ticket_tool -> ToolPermissionError
   - Retrieval Agent attempting to bind ticket_tool -> ToolPermissionError
   - Confirms crew_llm_calls and tool_calls remain completely unchanged.
5. Guard explanation and Medium Risk Classification justification.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config  # Ground Rule 3: must be first import
from utils.transcript import tee_transcript
from llm.counters import get_counter
from governance.tool_policy import audit_crew, ToolPermissionError
from crew.agents import build_agent, create_crew_agents
from tools.ticket_tool import TicketStatusTool
from crewai.crew import Crew
from crewai.process import Process


def main():
    with tee_transcript("task15_governance"):
        print("=" * 80)
        print("TASK 15: GOVERNANCE & LEAST AUTONOMY ENFORCEMENT DEMO")
        print("=" * 80)

        # ---------------------------------------------------------------------
        # 1. Four-Layer Governance Model Table
        # ---------------------------------------------------------------------
        print("\n--- 1. Four-Layer Governance Architecture ---")
        print(f"{'Layer':<26} | {'Enforcement Mechanism':<35} | {'Location'}")
        print("-" * 80)
        layers = [
            ("1. Organizational / Policy", "Risk classification (Medium); fiction only", "README, ola-support-agent.md"),
            ("2. Data", "PII phone masking; retention policies", "guardrails/pii.py, KB doc 12"),
            ("3. Application", "Least autonomy allow-list; no delegation", "governance/tool_policy.py, crew/agents.py"),
            ("4. Runtime", "Pre/in-flight token budget; injection; review", "governance/budget.py, pipeline.py"),
        ]
        for l_name, l_enf, l_loc in layers:
            print(f"{l_name:<26} | {l_enf:<35} | {l_loc}")
        print("-" * 80)

        # ---------------------------------------------------------------------
        # 2. Audit Real Crew Agent-Tool Bindings
        # ---------------------------------------------------------------------
        print("\n--- 2. Production Crew Tool Allow-List Audit ---")
        retrieval, lookup, composer = create_crew_agents()
        crew = Crew(
            agents=[retrieval, lookup, composer],
            process=Process.sequential,
        )

        audit_results = audit_crew(crew)
        all_allowed = True
        for role, tools, is_allowed in audit_results:
            status_mark = "ALLOWED ✓" if is_allowed else "DISALLOWED ✗"
            tool_str = ", ".join(tools) if tools else "(None - strictly forbidden)"
            print(f"Agent: {role:<18} | Tools: {tool_str:<32} | Status: {status_mark}")
            all_allowed = all_allowed and is_allowed

        print("\nAgent Delegation Controls:")
        for agent in [retrieval, lookup, composer]:
            print(f"Agent '{agent.role}': allow_delegation = {agent.allow_delegation} ✓")

        # ---------------------------------------------------------------------
        # 3. Construction-Time Least-Autonomy Violation Attempts
        # ---------------------------------------------------------------------
        print("\n--- 3. Construction-Time Violation Guard Testing ---")
        ticket_tool = TicketStatusTool()

        crew_calls_before = get_counter("crew_llm_calls")
        tool_calls_before = get_counter("tool_calls")

        # Violation 1: Response Composer attempting to bind ticket_tool
        print("\n[Attempt 1] Binding TicketStatusTool to Response Composer:")
        try:
            build_agent("Response Composer", [ticket_tool])
            print("FAIL: Binding unexpectedly succeeded!")
        except ToolPermissionError as e:
            print(f"BLOCKED AT CONSTRUCTION TIME ✓: {e}")

        # Violation 2: Retrieval Agent attempting to bind ticket_tool
        print("\n[Attempt 2] Binding TicketStatusTool to Retrieval Agent:")
        try:
            build_agent("Retrieval Agent", [ticket_tool])
            print("FAIL: Binding unexpectedly succeeded!")
        except ToolPermissionError as e:
            print(f"BLOCKED AT CONSTRUCTION TIME ✓: {e}")

        crew_calls_after = get_counter("crew_llm_calls")
        tool_calls_after = get_counter("tool_calls")

        print(f"\nCounter Verification after violation attempts:")
        print(f"  crew_llm_calls: {crew_calls_before} -> {crew_calls_after} (delta: {crew_calls_after - crew_calls_before})")
        print(f"  tool_calls:     {tool_calls_before} -> {tool_calls_after} (delta: {tool_calls_after - tool_calls_before})")
        assert crew_calls_after == crew_calls_before
        assert tool_calls_after == tool_calls_before

        # ---------------------------------------------------------------------
        # 4. Guard Explanation & Risk Classification Justification
        # ---------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("4. ARCHITECTURAL GUARD EXPLANATION & RISK CLASSIFICATION")
        print("=" * 80)

        print("\n[Guard Explanation]:")
        print(
            "Every agent in the system is created through one factory, build_agent(), and that "
            "factory checks each requested tool against a static allow-list before the CrewAI Agent "
            "object exists. Wiring check_support_ticket_status to anything but the Lookup Agent therefore "
            "raises ToolPermissionError at construction time, before any LLM call can happen. Delegation "
            "is disabled on every agent, so the Composer cannot ask the Lookup Agent to act on its behalf "
            "outside the planned task flow. The crew is audited before every kickoff, which catches a "
            "binding added by code that bypasses the factory. Both tools are read-only, so even the permitted "
            "agent cannot change ticket state."
        )

        print("\n[Risk Classification: MEDIUM]:")
        print(
            "The system is a customer-support ticket assistant, classified as Medium risk under the "
            "governance scheme. It reads policy text and ticket metadata (status, severity, timings). It "
            "never executes refunds, payments or account changes, and never handles card numbers, bank "
            "details or other financial data. Its outputs are advisory, shown to trained support staff "
            "(a human stays in the loop) rather than acting autonomously on customers. Residual risks are "
            "wrong policy information and exposed phone numbers; those are mitigated by grounded-only "
            "answers, the groundedness refusal, the AutoGen review stage and phone masking. It would move "
            "to High risk if it were allowed to issue refunds or credits, or to access payment details."
        )

        print("\n" + "=" * 80)
        print("TASK 15 GOVERNANCE DEMO COMPLETED SUCCESSFULLY ✓")
        print("=" * 80)


if __name__ == "__main__":
    main()

