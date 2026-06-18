"""Symptom Collector node — hybrid: emergency detection + LLM conversation + deterministic extraction."""

from __future__ import annotations

import json
import logging

from synapse.extractors.emergency_detector import detect_emergency
from synapse.llm.client import call_llm_json
from synapse.prompts.templates import SYMPTOM_COLLECTOR_SYSTEM_PROMPT
from synapse.state import SessionState, Symptom
from synapse.tools.registry import get_tools_for_node

logger = logging.getLogger(__name__)


def symptom_collector(state: SessionState) -> dict:
    """Collect symptoms through conversation. Emergency detection runs FIRST."""

    # PART A: Emergency detection — DETERMINISTIC, runs before LLM
    last_message = _get_last_patient_message(state)
    if last_message:
        emergency = detect_emergency(last_message)
        if emergency.detected:
            return {
                "emergency_detected": True,
                "emergency_type": emergency.emergency_type,
                "last_checkpoint_node": "symptom_collector",
            }

    # PART B: LLM conversation for symptom collection
    extracted_symptoms = state.get("extracted_symptoms", [])
    recent_messages = _get_recent_messages(state, count=3)

    prompt = SYMPTOM_COLLECTOR_SYSTEM_PROMPT.format(
        extracted_symptoms=json.dumps(
            [{"name": s.name, "severity": s.severity, "duration": s.duration} for s in extracted_symptoms],
            indent=2,
        ) if extracted_symptoms else "None yet",
        recent_messages=json.dumps(recent_messages, indent=2) if recent_messages else "None yet",
    )

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": last_message or "Patient has joined. Begin symptom collection."},
    ]

    tools = get_tools_for_node("symptom_collector")

    try:
        result = call_llm_json(
            messages,
            tools=tools,
            required_fields=["message"],
        )
    except Exception as e:
        logger.error("Symptom collector LLM failed: %s", e)
        result = {"message": "Can you tell me more about what you're experiencing?", "sufficient_data": False}

    # PART C: Handle tool calls from LLM
    new_symptoms = list(extracted_symptoms)
    sufficient = state.get("sufficient_data", False)

    # Check if LLM indicated sufficient data
    if result.get("sufficient_data"):
        sufficient = True

    turn_count = state.get("turn_count", 0) + 1

    return {
        "patient_facing_message": result.get("message", ""),
        "extracted_symptoms": new_symptoms,
        "sufficient_data": sufficient,
        "turn_count": turn_count,
        "last_checkpoint_node": "symptom_collector",
    }


def _get_last_patient_message(state: SessionState) -> str:
    """Extract the last patient message from conversation history.

    Handles both LangChain BaseMessage objects (type="human") and plain dicts (role="user").
    """
    messages = state.get("messages", [])
    for msg in reversed(messages):
        # LangChain BaseMessage: HumanMessage has type="human"
        if hasattr(msg, "type") and msg.type == "human":
            return msg.content if hasattr(msg, "content") else ""
        # LangChain BaseMessage with role attribute
        if hasattr(msg, "role") and msg.role == "user":
            return msg.content if hasattr(msg, "content") else ""
        # Plain dict
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def _get_recent_messages(state: SessionState, count: int = 3) -> list[dict]:
    """Get the last N messages as dicts.

    Handles both LangChain BaseMessage objects and plain dicts.
    """
    messages = state.get("messages", [])
    recent = []
    for msg in messages[-count:]:
        if hasattr(msg, "type") and msg.type in ("human", "ai", "system"):
            role = "user" if msg.type == "human" else msg.type
            recent.append({"role": role, "content": msg.content if hasattr(msg, "content") else ""})
        elif hasattr(msg, "role"):
            recent.append({"role": msg.role, "content": msg.content if hasattr(msg, "content") else ""})
        elif isinstance(msg, dict):
            recent.append(msg)
    return recent
