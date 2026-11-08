"""FastAPI application for Ola Support Agent (Section 15.2).

Endpoints:
- GET /health: returns HealthResponse {"status": "ok", "mock_llm": true}
- POST /ask: executes pipeline.answer() and returns AskResponse
- POST /add-document: chunks doc with deployed strategy, upserts to DEPLOYED_COLLECTION,
                      clears response cache, and returns AddDocumentResponse.
                      Note: Updates the vector index only; does not write to kb/.
- WS /ws/chat: real-time multi-turn chat over WebSockets. Survives abrupt client disconnects.
"""

import config  # Ground Rule 3: must be first import
import uuid
import time
from contextlib import asynccontextmanager
from typing import Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse

try:
    from websockets.exceptions import ConnectionClosed
except ImportError:
    ConnectionClosed = WebSocketDisconnect

from api.models import (
    AskRequest,
    AskResponse,
    AddDocumentRequest,
    AddDocumentResponse,
    HealthResponse,
)
from api.logging_jsonl import log_request, log_ws_event
from guardrails.pii import mask_pii
from rag.chunking import chunk_sentence_based
from rag.index import index_chunks, get_or_create_collection
from rag.embedder import get_embedder
from crew.agents import create_crew_agents
from governance.cache import cache
from llm.counters import get_counter
import pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up embedder, Chroma collection, and crew factory once at startup."""
    get_embedder()
    get_or_create_collection(config.DEPLOYED_COLLECTION)
    create_crew_agents()
    yield


app = FastAPI(title="Ola Support Agent API", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    t0 = time.perf_counter()
    trace_id = str(uuid.uuid4())
    lat = round((time.perf_counter() - t0) * 1000, 2)
    log_request({
        "trace_id": trace_id,
        "channel": "http",
        "endpoint": "/health",
        "session_id": "system",
        "request_text": "health check",
        "status": "ok",
        "intent": "health",
        "cache_hit": False,
        "guardrail_flags": [],
        "tokens_used": 0,
        "cost_inr": 0.0,
        "llm_calls": 0,
        "tool_calls": 0,
        "latency_ms": lat,
        "timings_ms": {"total": lat},
    })
    return HealthResponse(status="ok", mock_llm=True)


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """Execute end-to-end support pipeline on user question."""
    crew_0 = get_counter("crew_llm_calls")
    rev_0 = get_counter("review_llm_calls")
    tool_0 = get_counter("tool_calls")

    res = await pipeline.answer(session_id=request.session_id, question=request.question)

    crew_delta = get_counter("crew_llm_calls") - crew_0
    rev_delta = get_counter("review_llm_calls") - rev_0
    tool_delta = get_counter("tool_calls") - tool_0

    log_request({
        "trace_id": res.trace_id,
        "channel": "http",
        "endpoint": "/ask",
        "session_id": request.session_id,
        "request_text": res.masked_question,
        "status": res.status,
        "intent": res.intent,
        "cache_hit": res.cache_hit,
        "guardrail_flags": res.guardrail_flags,
        "tokens_used": res.tokens_used,
        "cost_inr": res.cost_inr,
        "llm_calls": crew_delta + rev_delta,
        "tool_calls": tool_delta,
        "latency_ms": res.latency_ms,
        "timings_ms": res.timings_ms,
    })

    return AskResponse.from_pipeline_result(res)


@app.post("/add-document", response_model=AddDocumentResponse)
async def add_document(request: AddDocumentRequest):
    """Add a document to the deployed vector collection.
    
    Chunks with the deployed sentence strategy, embeds, upserts into DEPLOYED_COLLECTION,
    and invalidates the response cache. Updates vector index only; does not write to kb/.
    """
    t0 = time.perf_counter()
    trace_id = str(uuid.uuid4())
    chunks = chunk_sentence_based(doc_id=request.doc_id, title=request.title, text=request.text)
    chunks_added = index_chunks(config.DEPLOYED_COLLECTION, chunks)

    # Invalidate response cache
    cache.clear()

    lat = round((time.perf_counter() - t0) * 1000, 2)
    masked_text = mask_pii(request.text)[0]
    log_request({
        "trace_id": trace_id,
        "channel": "http",
        "endpoint": "/add-document",
        "session_id": "system",
        "request_text": masked_text,
        "status": "ok",
        "intent": "add_document",
        "doc_id": request.doc_id,
        "chunks_added": chunks_added,
        "cache_hit": False,
        "guardrail_flags": [],
        "tokens_used": 0,
        "cost_inr": 0.0,
        "llm_calls": 0,
        "tool_calls": 0,
        "latency_ms": lat,
        "timings_ms": {"total": lat},
    })

    return AddDocumentResponse(
        doc_id=request.doc_id,
        chunks_added=chunks_added,
        collection=config.DEPLOYED_COLLECTION,
        cache_cleared=True,
    )


@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    """WebSocket endpoint for real-time multi-turn support chat."""
    await ws.accept()
    session_id = ws.query_params.get("session_id") or str(uuid.uuid4())
    conn_trace = str(uuid.uuid4())
    turns_completed = 0
    ws_start = time.perf_counter()
    log_ws_event("ws_connect", session_id=session_id, trace_id=conn_trace, latency_ms=0.0)

    try:
        while True:
            try:
                data = await ws.receive_json()
            except WebSocketDisconnect:
                raise
            except Exception as e:
                # Connection closed or error receiving
                if isinstance(e, (WebSocketDisconnect, ConnectionClosed)) or "closed" in str(e).lower():
                    raise
                await ws.send_json({"status": "error", "answer": f"Malformed payload: {str(e)}"})
                continue

            question = data.get("question", "") if isinstance(data, dict) else str(data)

            crew_0 = get_counter("crew_llm_calls")
            rev_0 = get_counter("review_llm_calls")
            tool_0 = get_counter("tool_calls")

            res = await pipeline.answer(session_id=session_id, question=question)
            turns_completed += 1

            crew_delta = get_counter("crew_llm_calls") - crew_0
            rev_delta = get_counter("review_llm_calls") - rev_0
            tool_delta = get_counter("tool_calls") - tool_0

            # Log turn with masked question
            log_request({
                "trace_id": res.trace_id,
                "channel": "ws",
                "endpoint": "/ws/chat",
                "session_id": session_id,
                "request_text": res.masked_question,
                "status": res.status,
                "intent": res.intent,
                "cache_hit": res.cache_hit,
                "guardrail_flags": res.guardrail_flags,
                "tokens_used": res.tokens_used,
                "cost_inr": res.cost_inr,
                "llm_calls": crew_delta + rev_delta,
                "tool_calls": tool_delta,
                "latency_ms": res.latency_ms,
                "timings_ms": res.timings_ms,
            })

            resp = AskResponse.from_pipeline_result(res)
            try:
                await ws.send_json(resp.model_dump())
            except (WebSocketDisconnect, ConnectionClosed, RuntimeError):
                # Client closed connection during send
                raise
    except (WebSocketDisconnect, ConnectionClosed, RuntimeError):
        pass
    except Exception:
        pass
    finally:
        total_ws_lat = round((time.perf_counter() - ws_start) * 1000, 2)
        log_ws_event(
            "ws_disconnect",
            session_id=session_id,
            trace_id=conn_trace,
            turns_completed=turns_completed,
            latency_ms=total_ws_lat,
        )

