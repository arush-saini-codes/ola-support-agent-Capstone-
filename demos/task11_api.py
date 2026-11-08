"""Task 11 Demo: FastAPI HTTP & WebSocket Endpoints (Section 15.2, 15.4).

Demonstrates:
a. GET /health
b. POST /ask "When do service credits expire?" -> AskResponse JSON
c. POST /add-document with "13_lost_item_recovery", then POST /ask testing retrieval of new doc,
   followed by index cleanup and verification of collection count back to 50.
d. WebSocket client A (session "ws_a"): 2 turns with session memory, then abrupt disconnect on turn 3.
e. WebSocket client B (session "ws_b") connects afterwards, receives normal response, and /health succeeds.
   Asserts "SERVER SURVIVED CLIENT DISCONNECT ✓".
f. Prints the last log lines showing the ws_disconnect event.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config  # Ground Rule 3: must be first import
import json
from fastapi.testclient import TestClient
from utils.transcript import tee_transcript
from api.main import app
from rag.index import get_or_create_collection
from governance.cache import cache


def main():
    with tee_transcript("task11_api"):
        print("=" * 80)
        print("TASK 11: FASTAPI HTTP & WEBSOCKET ENDPOINTS DEMO")
        print("=" * 80)

        with TestClient(app) as client:
            # -----------------------------------------------------------------
            # a. GET /health
            # -----------------------------------------------------------------
            print("\n--- a. GET /health ---")
            resp_health = client.get("/health")
            print(f"Status Code: {resp_health.status_code}")
            print(f"Response:    {json.dumps(resp_health.json(), indent=2)}")
            assert resp_health.status_code == 200
            assert resp_health.json()["status"] == "ok"
            assert resp_health.json()["mock_llm"] is True

            # -----------------------------------------------------------------
            # b. POST /ask
            # -----------------------------------------------------------------
            print("\n--- b. POST /ask ---")
            q_ask = "When do service credits expire?"
            resp_ask = client.post("/ask", json={"session_id": "api_demo_sess", "question": q_ask})
            print(f"Status Code: {resp_ask.status_code}")
            ask_json = resp_ask.json()
            print(f"AskResponse JSON:\n{json.dumps(ask_json, indent=2)}")
            assert resp_ask.status_code == 200
            assert ask_json["status"] == "ok"
            assert "08_service_credits" in ask_json["sources"]
            assert ask_json["approved"] is True

            # -----------------------------------------------------------------
            # c. POST /add-document & Retrieval Verification
            # -----------------------------------------------------------------
            print("\n--- c. POST /add-document & Retrieval Verification ---")
            doc_id = "13_lost_item_recovery"
            doc_title = "Lost Item Recovery Policy"
            doc_text = (
                "Items left behind in an Ola cab are logged within 2 hours by the driver. "
                "Lost property is held securely at the city hub for 30 days before disposal. "
                "Passengers may retrieve items with valid ID and payment of a Rs 99 handling fee."
            )
            resp_add = client.post(
                "/add-document",
                json={"doc_id": doc_id, "title": doc_title, "text": doc_text},
            )
            print(f"Status Code: {resp_add.status_code}")
            add_json = resp_add.json()
            print(f"AddDocumentResponse:\n{json.dumps(add_json, indent=2)}")
            assert resp_add.status_code == 200
            assert add_json["doc_id"] == doc_id
            assert add_json["chunks_added"] > 0
            assert add_json["cache_cleared"] is True

            # Ask question about the newly added policy
            q_new = "How long are lost items held at the city hub?"
            resp_ask_new = client.post("/ask", json={"session_id": "api_new_doc", "question": q_new})
            ask_new_json = resp_ask_new.json()
            print(f"\nAsk new doc response:\n{json.dumps(ask_new_json, indent=2)}")
            assert doc_id in ask_new_json["sources"], f"Expected {doc_id} in sources, got {ask_new_json['sources']}"

            # CLEANUP: delete doc chunks, clear cache, verify count back to 50
            collection = get_or_create_collection(config.DEPLOYED_COLLECTION)
            collection.delete(where={"doc_id": doc_id})
            cache.clear()
            final_count = collection.count()
            print(f"\nCleanup completed. Collection count back to: {final_count} (must be 50)")
            assert final_count == 50, f"Expected collection count 50 after cleanup, got {final_count}"

            # -----------------------------------------------------------------
            # d. WebSocket Client A (Multi-turn & Abrupt Disconnect)
            # -----------------------------------------------------------------
            print("\n--- d. WebSocket Client A (Multi-turn & Disconnect) ---")
            try:
                with client.websocket_connect("/ws/chat?session_id=ws_a") as ws_a:
                    # Turn 1
                    print("[ws_a] Sending Turn 1: 'Check OLA-1027'")
                    ws_a.send_json({"question": "Check OLA-1027"})
                    resp_t1 = ws_a.receive_json()
                    print(f"[ws_a] Turn 1 Full Answer:\n{resp_t1.get('answer')}\n")
                    assert resp_t1.get("status") == "ok"
                    assert resp_t1.get("ticket", {}).get("record_id") == "OLA-1027"

                    # Turn 2 (Memory context resolution over WS)
                    print("[ws_a] Sending Turn 2: 'What is its resolution SLA?'")
                    ws_a.send_json({"question": "What is its resolution SLA?"})
                    resp_t2 = ws_a.receive_json()
                    print(f"[ws_a] Turn 2 Full Answer:\n{resp_t2.get('answer')}\n")
                    assert resp_t2.get("status") == "ok"
                    assert "02_sla_by_severity" in resp_t2.get("sources", [])

                    # Turn 3: send and close immediately without reading reply
                    print("[ws_a] Sending Turn 3: 'Who handles it if open more than 7 days?' and closing socket abruptly...")
                    ws_a.send_json({"question": "Who handles it if open more than 7 days?"})
                    ws_a.close()
            except Exception as e:
                print(f"[ws_a] Disconnect handled gracefully: {e}")

            # Wait briefly (poll up to 5s) for the server to log client A's ws_disconnect
            import time
            log_path = config.LOGS_DIR / "requests.jsonl"
            ws_a_entries = []
            for _ in range(50):
                if log_path.exists():
                    all_entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").strip().splitlines()]
                    ws_a_entries = [e for e in all_entries if e.get("session_id") == "ws_a"]
                    if any(e.get("event") == "ws_disconnect" for e in ws_a_entries):
                        break
                time.sleep(0.1)

            print("\n--- Client A (session 'ws_a') Log Events Sequence ---")
            disc_entry = None
            for e in ws_a_entries:
                print(f"[{e.get('event') or e.get('endpoint')}]: {json.dumps(e, indent=2)}")
                if e.get("event") == "ws_disconnect":
                    disc_entry = e

            assert disc_entry is not None, "Expected ws_disconnect event for session 'ws_a'"
            turns_done = disc_entry.get("turns_completed")
            print(f"\nClient A Disconnect turns_completed = {turns_done}")
            if turns_done == 3:
                print("Explanation: turns_completed is 3 because Turn 3 completed pipeline execution before the socket close was detected during send_json.")
            else:
                print(f"Explanation: turns_completed is {turns_done} because the client socket disconnected before Turn 3 completed pipeline execution.")

            # -----------------------------------------------------------------
            # e. WebSocket Client B & Health Check (Proving Server Survived)
            # -----------------------------------------------------------------
            print("\n--- e. WebSocket Client B & Health Survival Check ---")
            with client.websocket_connect("/ws/chat?session_id=ws_b") as ws_b:
                print("[ws_b] Sending: 'When do service credits expire?'")
                ws_b.send_json({"question": "When do service credits expire?"})
                resp_b = ws_b.receive_json()
                print(f"[ws_b] Full Answer:\n{resp_b.get('answer')}\n")
                assert resp_b.get("status") == "ok"
                assert "08_service_credits" in resp_b.get("sources", [])

            resp_health2 = client.get("/health")
            assert resp_health2.status_code == 200
            assert resp_health2.json()["status"] == "ok"
            print("SERVER SURVIVED CLIENT DISCONNECT ✓")

            # -----------------------------------------------------------------
            # f. Print Last Log Lines Showing ws_disconnect
            # -----------------------------------------------------------------
            print("\n--- f. Last Log Lines in requests.jsonl Showing ws_disconnect ---")
            if log_path.exists():
                lines = log_path.read_text(encoding="utf-8").strip().splitlines()
                recent_lines = lines[-5:]
                for l in recent_lines:
                    print(l)

        print("\n" + "=" * 80)
        print("TASK 11 API DEMO COMPLETED SUCCESSFULLY ✓")
        print("=" * 80)


if __name__ == "__main__":
    main()

