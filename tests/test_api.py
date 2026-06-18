"""Tests for the FastAPI API endpoints."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from synapse.llm.client import set_llm_override


def _mock_llm(messages, tools=None):
    """Mock LLM for API tests."""
    for m in messages:
        if m.get("role") == "system":
            sys = m["content"].lower()
            if "welcome" in sys or "greet" in sys:
                return {"content": json.dumps({"message": "Hello! How can I help?", "next_action": "ask_symptoms"})}
            elif "symptom" in sys:
                return {"content": json.dumps({"message": "Tell me more.", "sufficient_data": True})}
            elif "explaining" in sys:
                return {"content": json.dumps({"message": "Please go to GP."})}
            elif "closing" in sys:
                return {"content": json.dumps({"closing_message": "Take care!"})}
    return {"content": json.dumps({"message": "OK"})}


@pytest.fixture(autouse=True)
def setup_mock():
    """Inject mock LLM for all API tests."""
    set_llm_override(_mock_llm)
    yield
    set_llm_override(None)


@pytest.fixture
def client():
    """Create test client."""
    from synapse.api.app import app
    return TestClient(app)


class TestHealthCheck:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "synapse-triage"


class TestSessionEndpoint:
    def test_get_session_not_found(self, client):
        response = client.get("/session/nonexistent")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "not_found"


class TestTriageWebSocket:
    def test_full_conversation(self, client):
        """Test a complete triage conversation via WebSocket."""
        with client.websocket_connect("/ws/triage") as ws:
            # Receive session init
            init = ws.receive_json()
            assert init["type"] == "session_init"
            session_id = init["session_id"]
            assert session_id

            # Send patient message
            ws.send_json({
                "type": "message",
                "content": "I have a headache",
            })

            # Receive response
            response = ws.receive_json()
            assert response["type"] == "response"
            assert response["content"]
            assert response["session_id"] == session_id
            assert "turn_count" in response["metadata"]

    def test_emergency_detection(self, client):
        """Test emergency detection via WebSocket."""
        with client.websocket_connect("/ws/triage") as ws:
            init = ws.receive_json()
            assert init["type"] == "session_init"

            # Send emergency message
            ws.send_json({
                "type": "message",
                "content": "I have chest pain and can't breathe",
            })

            response = ws.receive_json()
            assert response["type"] == "emergency"
            assert response["metadata"]["emergency_detected"] is True
            assert response["metadata"]["emergency_type"] == "cardiac"

    def test_empty_message_returns_error(self, client):
        """Empty messages should return error."""
        with client.websocket_connect("/ws/triage") as ws:
            ws.receive_json()  # init

            ws.send_json({"type": "message", "content": ""})
            response = ws.receive_json()
            assert response["type"] == "error"

    def test_invalid_json_returns_error(self, client):
        """Invalid JSON should return error."""
        with client.websocket_connect("/ws/triage") as ws:
            ws.receive_json()  # init

            ws.send_text("not json")
            response = ws.receive_json()
            assert response["type"] == "error"
            assert "Invalid JSON" in response["message"]
