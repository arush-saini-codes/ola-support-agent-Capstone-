"""Tests for FastAPI HTTP and WebSocket endpoints (Phase 9).

Covers:
1. GET /health status and schema.
2. POST /ask schema and status.
3. POST /add-document validation error on bad doc_id (422).
4. WebSocket round-trip.
5. Server survival after WebSocket abrupt disconnect.
6. Structured logging with trace_id and absence of raw phone PII.
7. Cleanup after /add-document tests.
"""

import config  # Ground Rule 3: must be first import
import json
import pytest
from fastapi.testclient import TestClient
from api.main import app
from api.models import HealthResponse, AskResponse, AddDocumentResponse
from rag.index import get_or_create_collection
from governance.cache import cache


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    """GET /health returns 200 with HealthResponse schema."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    model = HealthResponse(**data)
    assert model.status == "ok"
    assert model.mock_llm is True


def test_ask_endpoint_schema(client):
    """POST /ask returns 200 and validates against AskResponse schema."""
    payload = {"session_id": "test_sess_ask", "question": "When do service credits expire?"}
    resp = client.post("/ask", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    model = AskResponse(**data)
    assert model.status == "ok"
    assert model.trace_id is not None
    assert "08_service_credits" in model.sources
    assert model.tokens_used > 0
    assert model.cost_inr >= 0.0


def test_add_document_validation_error_on_bad_id(client):
    """POST /add-document with invalid doc_id pattern raises 422."""
    bad_payload = {
        "doc_id": "INVALID-DOC-ID!",  # violates ^[a-z0-9_]+$
        "title": "Bad Doc Title",
        "text": "This is a sufficiently long text for validation testing purposes.",
    }
    resp = client.post("/add-document", json=bad_payload)
    assert resp.status_code == 422


def test_add_document_success_and_cleanup(client):
    """POST /add-document upserts chunks, and cleanup restores collection count to 50."""
    doc_id = "test_custom_doc"
    payload = {
        "doc_id": doc_id,
        "title": "Custom Test Policy",
        "text": "Drivers must inspect the vehicle cabin at the end of every trip. Lost belongings are deposited safely.",
    }
    resp = client.post("/add-document", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    model = AddDocumentResponse(**data)
    assert model.doc_id == doc_id
    assert model.chunks_added > 0
    assert model.cache_cleared is True

    # Cleanup chunks
    collection = get_or_create_collection(config.DEPLOYED_COLLECTION)
    collection.delete(where={"doc_id": doc_id})
    cache.clear()
    assert collection.count() == 50


def test_websocket_roundtrip(client):
    """WS /ws/chat handles interactive message send and receive."""
    with client.websocket_connect("/ws/chat?session_id=test_ws_rt") as ws:
        ws.send_json({"question": "When do service credits expire?"})
        data = ws.receive_json()
        assert data.get("status") == "ok"
        assert "08_service_credits" in data.get("sources", [])


def test_websocket_survival_after_disconnect(client):
    """Server handles mid-session disconnect and survives for subsequent connections."""
    try:
        with client.websocket_connect("/ws/chat?session_id=test_ws_disc") as ws:
            ws.send_json({"question": "Check OLA-1027"})
            ws.receive_json()
            # Send and abruptly disconnect
            ws.send_json({"question": "What is its resolution SLA?"})
            ws.close()
    except Exception:
        pass

    # Reconnect another client to ensure server is healthy
    with client.websocket_connect("/ws/chat?session_id=test_ws_disc_after") as ws2:
        ws2.send_json({"question": "When do service credits expire?"})
        data = ws2.receive_json()
        assert data.get("status") == "ok"

    # Health check survives
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_logging_pii_defense_and_trace_id(client):
    """Requests log trace_id and mask phone numbers completely."""
    log_file = config.LOGS_DIR / "requests.jsonl"
    lines_before = log_file.read_text(encoding="utf-8").strip().splitlines() if log_file.exists() else []

    phone_num = "+91 98765 43210"
    resp = client.post(
        "/ask",
        json={"session_id": "test_pii_log", "question": f"Call me at {phone_num} about OLA-1027"},
    )
    assert resp.status_code == 200
    resp_data = resp.json()
    resp_trace = resp_data["trace_id"]

    lines_after = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines_after) > len(lines_before)

    last_entry = json.loads(lines_after[-1])
    assert last_entry["trace_id"] == resp_trace
    assert "[PHONE]" in last_entry["request_text"]
    assert "latency_ms" in last_entry
    assert "timings_ms" in last_entry
    assert last_entry["llm_calls"] < 15

    all_logs = log_file.read_text(encoding="utf-8")
    assert "9876543210" not in all_logs
    assert "98765 43210" not in all_logs

