"""Task 16 Demo: Response Cache with LRU Eviction & Invalidation (Section 14.4).

Demonstrates:
1. Clearing cache before execution.
2. 1st Request: "When do service credits expire?" -> Cache miss, executes crew + review.
3. 2nd Request: "  when do SERVICE credits expire  " -> Normalized match, cache hit.
4. Before/after table comparing crew_llm_calls, tool_calls, generation_calls, review_llm_calls,
   latency_ms, and cache_hit.
5. Invalidation via cache.clear() proving subsequent query is a miss again.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config  # Ground Rule 3: must be first import
from utils.transcript import tee_transcript
from llm.counters import get_counter
from governance.cache import cache
from pipeline import answer_sync


def main():
    with tee_transcript("task16_cache"):
        print("=" * 80)
        print("TASK 16: RESPONSE CACHE & INLINE RETRIEVAL OPTIMIZATION DEMO")
        print("=" * 80)

        # Step 1: Ensure clear baseline
        cache.clear()
        session_id = "cache_demo_session"

        q1 = "When do service credits expire?"
        q2 = "  when do SERVICE credits expire  "

        # ---------------------------------------------------------------------
        # 1st Request (Cache Miss)
        # ---------------------------------------------------------------------
        crew_0 = get_counter("crew_llm_calls")
        tool_0 = get_counter("tool_calls")
        gen_0 = get_counter("generation_calls")
        rev_0 = get_counter("review_llm_calls")

        res_1 = answer_sync(session_id, q1)

        crew_1 = get_counter("crew_llm_calls")
        tool_1 = get_counter("tool_calls")
        gen_1 = get_counter("generation_calls")
        rev_1 = get_counter("review_llm_calls")

        delta_crew_1 = crew_1 - crew_0
        delta_tool_1 = tool_1 - tool_0
        delta_gen_1 = gen_1 - gen_0
        delta_rev_1 = rev_1 - rev_0

        assert res_1.cache_hit is False

        # ---------------------------------------------------------------------
        # 2nd Request (Cache Hit on Normalized Query)
        # ---------------------------------------------------------------------
        crew_start_2 = get_counter("crew_llm_calls")
        tool_start_2 = get_counter("tool_calls")
        gen_start_2 = get_counter("generation_calls")
        rev_start_2 = get_counter("review_llm_calls")

        res_2 = answer_sync(session_id, q2)

        crew_end_2 = get_counter("crew_llm_calls")
        tool_end_2 = get_counter("tool_calls")
        gen_end_2 = get_counter("generation_calls")
        rev_end_2 = get_counter("review_llm_calls")

        delta_crew_2 = crew_end_2 - crew_start_2
        delta_tool_2 = tool_end_2 - tool_start_2
        delta_gen_2 = gen_end_2 - gen_start_2
        delta_rev_2 = rev_end_2 - rev_start_2

        assert res_2.cache_hit is True
        assert delta_crew_2 == 0
        assert delta_tool_2 == 0
        assert delta_gen_2 == 0
        assert delta_rev_2 == 0
        assert res_1.answer == res_2.answer

        # ---------------------------------------------------------------------
        # Print Before/After Table (Section 14.4 format)
        # ---------------------------------------------------------------------
        print("\n--- Section 14.4 Performance & Counter Comparison Table ---")
        header = f"{'Request':<22} | {'crew_llm':<8} | {'tool_calls':<10} | {'gen_calls':<9} | {'review_llm':<10} | {'tokens_used':<11} | {'latency_ms':<10} | {'cache_hit'}"
        print(header)
        print("-" * len(header))
        print(
            f"{'1st request (miss)':<22} | "
            f"+{delta_crew_1:<7} | "
            f"+{delta_tool_1:<9} | "
            f"+{delta_gen_1:<8} | "
            f"+{delta_rev_1:<9} | "
            f"{res_1.tokens_used:<11} | "
            f"{res_1.latency_ms:<10.1f} | "
            f"{res_1.cache_hit}"
        )
        print(
            f"{'2nd (same query - hit)':<22} | "
            f"+{delta_crew_2:<7} | "
            f"+{delta_tool_2:<9} | "
            f"+{delta_gen_2:<8} | "
            f"+{delta_rev_2:<9} | "
            f"{res_2.tokens_used:<11} | "
            f"{res_2.latency_ms:<10.1f} | "
            f"{res_2.cache_hit}"
        )
        print("-" * len(header))

        print(f"\n1st Answer:\n{res_1.answer}")
        print(f"\n2nd Answer:\n{res_2.answer}")
        print("\nAnswer Match Verification: IDENTICAL ✓")

        # ---------------------------------------------------------------------
        # Separate Invalidation Demo
        # ---------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("SEPARATE SECTION: CACHE INVALIDATION VIA cache.clear()")
        print("=" * 80)
        stats_before = cache.stats()
        print(f"Stats Before Invalidation: {stats_before}")

        cache.clear()
        stats_after = cache.stats()
        print(f"Stats After Invalidation:  {stats_after}")
        assert stats_after["entries"] == 0

        # Ask again -> must be a miss and counters must increase again
        crew_pre_inv = get_counter("crew_llm_calls")
        tool_pre_inv = get_counter("tool_calls")
        gen_pre_inv = get_counter("generation_calls")
        rev_pre_inv = get_counter("review_llm_calls")

        res_3 = answer_sync(session_id, q1)

        crew_post_inv = get_counter("crew_llm_calls")
        tool_post_inv = get_counter("tool_calls")
        gen_post_inv = get_counter("generation_calls")
        rev_post_inv = get_counter("review_llm_calls")

        d_crew_3 = crew_post_inv - crew_pre_inv
        d_tool_3 = tool_post_inv - tool_pre_inv
        d_gen_3 = gen_post_inv - gen_pre_inv
        d_rev_3 = rev_post_inv - rev_pre_inv

        print(f"Subsequent Ask Cache Hit Status: {res_3.cache_hit} (Successfully invalidated -> Miss ✓)")
        print(f"Post-Invalidation Counters Increase:")
        print(f"  crew_llm_calls delta:   +{d_crew_3} (expected > 0)")
        print(f"  tool_calls delta:       +{d_tool_3}")
        print(f"  generation_calls delta: +{d_gen_3}")
        print(f"  review_llm_calls delta: +{d_rev_3}")

        assert res_3.cache_hit is False
        assert d_crew_3 > 0
        assert d_rev_3 > 0

        # ---------------------------------------------------------------------
        # Separate LRU Eviction Demo
        # ---------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("SEPARATE SECTION: LRU CAPACITY EVICTION CHECK")
        print("=" * 80)
        from governance.cache import ResponseCache
        small_cache = ResponseCache(max_entries=2)
        small_cache.put("q1", {"status": "ok", "answer": "a1"})
        small_cache.put("q2", {"status": "ok", "answer": "a2"})
        print(f"Inserted q1, q2 into cache(cap=2): entries = {list(small_cache._store.keys())}")
        # Access q1 to make it most recently used
        small_cache.get("q1")
        # Put q3 -> should evict q2 (least recently used)
        small_cache.put("q3", {"status": "ok", "answer": "a3"})
        retained = list(small_cache._store.keys())
        print(f"Accessed q1, inserted q3 -> retained entries: {retained}")
        assert "q2" not in retained, "Expected q2 to be evicted"
        assert "q1" in retained and "q3" in retained
        print("LRU Eviction Order: Correctly evicted least recently used item ✓")

        print("\n" + "=" * 80)
        print("TASK 16 CACHE DEMO COMPLETED SUCCESSFULLY ✓")
        print("=" * 80)


if __name__ == "__main__":
    main()

