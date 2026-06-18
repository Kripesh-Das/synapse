"""FastAPI application for synapse Triage Agent.

Provides:
- WebSocket endpoint for real-time triage conversations
- REST endpoints for session management and health checks

Usage:
    uvicorn synapse.api.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from synapse.graph import build_graph
from synapse.state import SessionState

logger = logging.getLogger(__name__)

_graph = None


def get_graph():
    """Get or build the triage graph."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    logger.info("Starting synapse API server...")
    get_graph()
    yield
    logger.info("Shutting down synapse API server...")


app = FastAPI(
    title="synapse Triage Agent",
    description="Hospital AI Triage Assistant API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- REST Endpoints ---

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "synapse-triage"}


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session status."""
    try:
        from synapse.cache.session import get_cached_session
        cached = get_cached_session(session_id)
        if cached:
            return {"status": "active", "session": cached}
    except Exception:
        pass

    return {"status": "not_found", "session_id": session_id}


# --- WebSocket Endpoint ---

@app.websocket("/ws/triage")
async def triage_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time triage conversations.

    Protocol:
    Client sends JSON: {"type": "message", "content": "...", "session_id": "..."}
    Server sends JSON: {"type": "response", "content": "...", "session_id": "...", "metadata": {...}}

    Message types:
    - message: Patient message
    - response: Agent response
    - emergency: Emergency alert
    - error: Error message
    - session_init: New session started
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    messages = []
    graph = get_graph()

    logger.info("WebSocket connected: session=%s", session_id)

    # Send session init
    await websocket.send_json({
        "type": "session_init",
        "session_id": session_id,
        "message": "Connected to synapse triage assistant.",
    })

    try:
        while True:
            # Receive patient message
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON",
                })
                continue

            msg_type = data.get("type", "message")
            content = data.get("content", "")

            if not content:
                await websocket.send_json({
                    "type": "error",
                    "message": "Empty message",
                })
                continue

            # Add patient message
            messages.append({"role": "user", "content": content})

            # Build state
            state: SessionState = {
                "messages": list(messages),
                "extracted_symptoms": [],
                "source": "mobile",
                "language": data.get("language", "en"),
                "session_id": session_id,
            }

            # Run graph
            try:
                config = {"configurable": {"thread_id": session_id}}
                result = graph.invoke(state, config=config)
            except Exception as e:
                logger.error("Graph execution failed: %s", e)
                await websocket.send_json({
                    "type": "error",
                    "message": "Processing error. Please try again.",
                })
                continue

            # Extract response
            response_content = result.get("patient_facing_message", "")
            emergency = result.get("emergency_detected", False)
            triage_result = result.get("triage_result")

            # Add agent response to history
            messages.append({"role": "assistant", "content": response_content})

            # Build response
            response = {
                "type": "emergency" if emergency else "response",
                "content": response_content,
                "session_id": session_id,
                "metadata": {
                    "turn_count": result.get("turn_count", 0),
                    "sufficient_data": result.get("sufficient_data", False),
                    "emergency_detected": emergency,
                    "last_node": result.get("last_checkpoint_node", ""),
                },
            }

            if triage_result and hasattr(triage_result, "score"):
                response["metadata"]["triage_score"] = triage_result.score
                response["metadata"]["department"] = triage_result.department

            if result.get("emergency_type"):
                response["metadata"]["emergency_type"] = result["emergency_type"]

            await websocket.send_json(response)

            # If session closed, break
            if result.get("last_checkpoint_node") == "session_close":
                break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: session=%s", session_id)
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        try:
            await websocket.send_json({
                "type": "error",
                "message": "Connection error",
            })
        except Exception:
            pass
