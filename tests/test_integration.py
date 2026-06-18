"""Integration test — runs the full graph flow with a mock LLM.

Tests the complete triage pipeline from session_manager through session_close.
No real LLM calls — uses deterministic mock responses.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

from synapse.graph import build_graph
from synapse.llm.client import set_llm_override
from synapse.state import SessionState, Symptom, TriageResult


# --- Mock LLM responses for each node ---

MOCK_RESPONSES = {
    "welcome_agent": {
        "content": json.dumps({
            "message": "Welcome to synapse. How can I help you today?",
            "next_action": "ask_symptoms",
        }),
    },
    "symptom_collector": {
        "content": json.dumps({
            "message": "I understand you have chest pain. Can you tell me how long you've had this and how severe it is on a scale of 1-10?",
            "sufficient_data": False,
            "missing_fields": ["duration", "severity"],
        }),
    },
    "summary_generator": {
        "content": json.dumps({
            "message": "Based on your symptoms, please proceed to the Cardiology department. Estimated wait time is approximately 45 minutes. If your condition worsens, please alert staff immediately.",
            "include_map": True,
            "include_preparation": True,
        }),
    },
    "session_close": {
        "content": json.dumps({
            "closing_message": "Thank you for using synapse. Take care and we hope you feel better soon.",
            "feedback_request": "How was your experience? (1-5)",
        }),
    },
}


def _mock_llm(messages: list[dict], tools: list[dict] | None = None) -> dict[str, Any]:
    """Mock LLM that returns appropriate responses based on the system prompt."""
    system_msg = ""
    for m in messages:
        if m.get("role") == "system":
            system_msg = m.get("content", "")
            break

    if "welcome" in system_msg.lower() or "greet" in system_msg.lower():
        return MOCK_RESPONSES["welcome_agent"]
    elif "collecting symptoms" in system_msg.lower() or "symptom" in system_msg.lower():
        return MOCK_RESPONSES["symptom_collector"]
    elif "explaining the next steps" in system_msg.lower() or "summary" in system_msg.lower():
        return MOCK_RESPONSES["summary_generator"]
    elif "closing message" in system_msg.lower() or "goodbye" in system_msg.lower():
        return MOCK_RESPONSES["session_close"]
    else:
        return {"content": json.dumps({"message": "I'm here to help."})}


class TestFullTriageFlow:
    """End-to-end test: patient with chest pain goes through full triage."""

    def setup_method(self):
        set_llm_override(_mock_llm)

    def teardown_method(self):
        set_llm_override(None)

    def test_full_flow_non_urgent(self):
        """Patient with moderate headache — should complete full flow."""
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())

        initial_state: SessionState = {
            "messages": [
                {"role": "user", "content": "I have a headache for 2 days"},
            ],
            "extracted_symptoms": [
                Symptom(name="headache", duration="2 days", severity=5),
            ],
            "source": "web",
            "language": "en",
        }

        config = {"configurable": {"thread_id": "test-non-urgent"}}
        result = graph.invoke(initial_state, config=config)

        # Should have gone through all nodes
        assert result.get("last_checkpoint_node") == "session_close"
        assert result.get("patient_facing_message")
        assert result.get("clinician_summary")
        assert result.get("triage_result") is not None

        # Triage should be non-urgent (score > 2)
        triage: TriageResult = result["triage_result"]
        assert triage.score > 2
        assert triage.department  # Should have a department assigned

    def test_full_flow_emergency(self):
        """Patient with chest pain — should trigger emergency handler."""
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())

        initial_state: SessionState = {
            "messages": [
                {"role": "user", "content": "I have severe chest pain"},
            ],
            "extracted_symptoms": [],
            "source": "kiosk",
            "language": "en",
        }

        config = {"configurable": {"thread_id": "test-emergency"}}
        result = graph.invoke(initial_state, config=config)

        # Emergency detected → emergency_handler → session_close
        assert result.get("emergency_detected") is True
        assert result.get("emergency_type") == "cardiac"
        assert result.get("last_checkpoint_node") == "session_close"
        # session_close overwrites patient_facing_message with closing message (correct)
        # The emergency message was shown during emergency_handler step
        assert result.get("patient_facing_message")  # Has a closing message

    def test_full_flow_urgent(self):
        """Patient with critical symptoms — should go through urgent_notification."""
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())

        initial_state: SessionState = {
            "messages": [
                {"role": "user", "content": "I think I'm having a stroke"},
            ],
            "extracted_symptoms": [
                Symptom(name="stroke", severity=10, onset="sudden"),
            ],
            "source": "web",
            "language": "en",
        }

        config = {"configurable": {"thread_id": "test-urgent"}}
        result = graph.invoke(initial_state, config=config)

        # Should detect emergency (stroke is in keyword list)
        assert result.get("emergency_detected") is True
        assert result.get("emergency_type") == "neurological"


class TestTriageScoring:
    """Test triage scoring through the graph."""

    def setup_method(self):
        set_llm_override(_mock_llm)

    def teardown_method(self):
        set_llm_override(None)

    def test_critical_score(self):
        """Chest pain should score 1 (critical)."""
        from synapse.triage.rules_engine import TriageRulesEngine

        engine = TriageRulesEngine()
        result = engine.compute_triage([Symptom(name="chest pain", severity=9)])
        assert result.score == 1
        assert result.department == "ED"

    def test_moderate_score(self):
        """Headache with moderate severity should score 3-5."""
        from synapse.triage.rules_engine import TriageRulesEngine

        engine = TriageRulesEngine()
        result = engine.compute_triage([Symptom(name="headache", severity=5)])
        assert result.score >= 3

    def test_elderly_adjustment(self):
        """Elderly patient should get upgraded urgency."""
        from synapse.triage.rules_engine import TriageRulesEngine

        engine = TriageRulesEngine()
        result = engine.compute_triage(
            [Symptom(name="dizziness", severity=4)],
            age=75,
        )
        # Without age: score 5, with age: score 4
        assert result.score == 4


class TestEmergencyDetection:
    """Test emergency detection through the graph."""

    def setup_method(self):
        set_llm_override(_mock_llm)

    def teardown_method(self):
        set_llm_override(None)

    def test_cardiac_triggers_emergency(self):
        """'chest pain' should trigger cardiac emergency."""
        from synapse.extractors.emergency_detector import detect_emergency

        result = detect_emergency("I have chest pain")
        assert result.detected is True
        assert result.emergency_type == "cardiac"

    def test_neurological_triggers_emergency(self):
        """'stroke' should trigger neurological emergency."""
        from synapse.extractors.emergency_detector import detect_emergency

        result = detect_emergency("I think it's a stroke")
        assert result.detected is True
        assert result.emergency_type == "neurological"

    def test_no_emergency_for_mild_symptoms(self):
        """Mild symptoms should not trigger emergency."""
        from synapse.extractors.emergency_detector import detect_emergency

        result = detect_emergency("I have a mild headache")
        assert result.detected is False


class TestToolRegistry:
    """Test tool execution."""

    def test_end_conversation(self):
        from synapse.tools.registry import execute_tool

        result = execute_tool("end_conversation", {"reason": "patient_request"})
        assert result["status"] == "ended"

    def test_request_human(self):
        from synapse.tools.registry import execute_tool

        result = execute_tool("request_human", {"reason": "patient_request", "urgency": "medium"})
        assert result["status"] == "queued"

    def test_unknown_tool(self):
        from synapse.tools.registry import execute_tool

        result = execute_tool("nonexistent_tool", {})
        assert "error" in result

    def test_progressive_disclosure(self):
        from synapse.tools.registry import get_tools_for_node

        welcome_tools = get_tools_for_node("welcome_agent")
        assert len(welcome_tools) == 3  # Only conversation tools

        symptom_tools = get_tools_for_node("symptom_collector")
        assert len(symptom_tools) == 10  # Conversation + symptom tools

        summary_tools = get_tools_for_node("summary_generator")
        assert len(summary_tools) == 6  # Conversation + knowledge tools


class TestGraphStructure:
    """Test graph compilation and node connectivity."""

    def test_graph_compiles(self):
        graph = build_graph()
        assert graph is not None

    def test_all_nodes_present(self):
        graph = build_graph()
        nodes = graph.get_graph().nodes.keys()
        expected = [
            "__start__", "session_manager", "welcome_agent",
            "symptom_collector", "emergency_handler", "triage_engine",
            "urgent_notification", "summary_generator", "data_persistence",
            "session_close", "error_handler", "__end__",
        ]
        for node in expected:
            assert node in nodes, f"Missing node: {node}"
