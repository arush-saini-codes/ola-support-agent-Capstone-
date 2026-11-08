"""Smoke test running real uvicorn server in background, testing with httpx (Section 15.4)."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config  # Ground Rule 3: must be first import
import subprocess
import time
import httpx
import json


def main():
    print("=" * 80)
    print("REAL SERVER SMOKE TEST (Uvicorn + httpx against http://127.0.0.1:8000)")
    print("=" * 80)

    cmd = [
        str(ROOT_DIR / ".venv" / "Scripts" / "uvicorn.exe"),
        "api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]

    print(f"Starting server: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        # Wait for server to bind and warm up
        base_url = "http://127.0.0.1:8000"
        client = httpx.Client(base_url=base_url, timeout=60.0)

        ready = False
        for attempt in range(20):
            try:
                r = client.get("/health")
                if r.status_code == 200:
                    ready = True
                    break
            except Exception:
                time.sleep(0.5)

        if not ready:
            print("ERROR: Server failed to start within timeout.")
            return

        print("\n--- 1. GET /health Response ---")
        health_resp = client.get("/health")
        print(f"Status Code: {health_resp.status_code}")
        print(f"Response Body:\n{json.dumps(health_resp.json(), indent=2)}")

        print("\n--- 2. POST /ask Response ---")
        ask_resp = client.post(
            "/ask",
            json={"session_id": "smoke_test_sess", "question": "When do service credits expire?"},
        )
        print(f"Status Code: {ask_resp.status_code}")
        print(f"Response Body:\n{json.dumps(ask_resp.json(), indent=2)}")

        print("\nREAL SERVER SMOKE TEST COMPLETED SUCCESSFULLY ✓")

    finally:
        print("\nStopping uvicorn server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Server stopped.")


if __name__ == "__main__":
    main()
