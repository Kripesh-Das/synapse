"""Symptom Collector node — hybrid: emergency detection + NER + LLM conversation."""

from __future__ import annotations

import json
import logging

from synapse.extractors.emergency_detector import detect_emergency
from synapse.extractors.medical_ner import get_ner
from synapse.llm.client import call_llm_json
from synapse.prompts.templates import SYMPTOM_COLLECTOR_SYSTEM_PROMPT
from synapse.state import SessionState, Symptom
from synapse.tools.registry import get_tools_for_node

logger = logging.getLogger(__name__)


def symptom_collector(state: SessionState) -> dict:
    """Collect symptoms through NER extraction + LLM conversation.

    Flow:
    1. Emergency detection — DETERMINISTIC, runs before LLM
    2. NER extraction — DETERMINISTIC, extracts symptoms from patient message
    3. LLM conversation — fills gaps, asks follow-up questions
    """

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

    # PART B: NER extraction — DETERMINISTIC, extracts structured data
    extracted_symptoms = list(state.get("extracted_symptoms", []))
    extracted_entities = dict(state.get("extracted_entities", {}))
    ner_context = {}

    if last_message:
        ner = get_ner()
        ner_result = ner.extract(last_message)

        # Merge NER-extracted symptoms into extracted_symptoms
        existing_names = {s.name.lower() for s in extracted_symptoms}
        for symptom_entity in ner_result.symptoms:
            if not symptom_entity.negated and symptom_entity.text.lower() not in existing_names:
                # Create Symptom from NER entity
                new_symptom = Symptom(name=symptom_entity.text.lower())

                # Try to link body part
                if ner_result.body_parts:
                    new_symptom.location = ner_result.body_parts[0].text

                # Try to link duration
                if ner_result.duration:
                    new_symptom.duration = ner_result.duration[0].text

                # Try to link severity
                if ner_result.severity:
                    severity_text = ner_result.severity[0].text
                    # Parse numeric severity (e.g., "7/10" -> 7)
                    severity_num = _parse_severity(severity_text)
                    if severity_num:
                        new_symptom.severity = severity_num

                # Try to link onset
                if ner_result.onset:
                    new_symptom.onset = ner_result.onset[0].text

                extracted_symptoms.append(new_symptom)
                existing_names.add(symptom_entity.text.lower())

        # Store NER entities in extracted_entities
        ner_entities = ner_result.to_dict()
        for key, values in ner_entities.items():
            if values and key not in extracted_entities:
                extracted_entities[key] = values

        # Build NER context for LLM
        ner_context = {
            "symptoms": [e.text for e in ner_result.symptoms if not e.negated],
            "body_parts": [e.text for e in ner_result.body_parts if not e.negated],
            "medications": [e.text for e in ner_result.medications],
            "duration": [e.text for e in ner_result.duration],
            "severity": [e.text for e in ner_result.severity],
            "onset": [e.text for e in ner_result.onset],
        }

    # PART C: LLM conversation for symptom collection
    recent_messages = _get_recent_messages(state, count=3)

    prompt = SYMPTOM_COLLECTOR_SYSTEM_PROMPT.format(
        extracted_symptoms=json.dumps(
            [{"name": s.name, "severity": s.severity, "duration": s.duration,
              "location": s.location, "onset": s.onset} for s in extracted_symptoms],
            indent=2,
        ) if extracted_symptoms else "None yet",
        recent_messages=json.dumps(recent_messages, indent=2) if recent_messages else "None yet",
        ner_context=json.dumps(ner_context, indent=2) if ner_context else "None",
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

    # PART D: Handle tool calls from LLM
    sufficient = state.get("sufficient_data", False)

    # Check if LLM indicated sufficient data
    if result.get("sufficient_data"):
        sufficient = True

    turn_count = state.get("turn_count", 0) + 1

    return {
        "patient_facing_message": result.get("message", ""),
        "extracted_symptoms": extracted_symptoms,
        "extracted_entities": extracted_entities,
        "sufficient_data": sufficient,
        "turn_count": turn_count,
        "last_checkpoint_node": "symptom_collector",
    }


def _parse_severity(text: str) -> int:
    """Parse severity text into a 1-10 integer.

    Handles formats like:
    - "7/10" -> 7
    - "7 out of 10" -> 7
    - "severe" -> 8
    - "mild" -> 3
    """
    text = text.lower().strip()

    # Try numeric formats: "7/10", "7 out of 10"
    match = re.match(r"(\d+)\s*/\s*(\d+)", text)
    if match:
        return min(10, max(1, int(match.group(1))))

    match = re.match(r"(\d+)\s+out\s+of\s+(\d+)", text)
    if match:
        return min(10, max(1, int(match.group(1))))

    # Word-based severity mapping
    severity_map = {
        "mild": 3,
        "moderate": 5,
        "severe": 8,
        "extreme": 9,
        "terrible": 9,
        "horrible": 9,
        "intense": 8,
        "sharp": 7,
        "dull": 4,
        "throbbing": 7,
        "stabbing": 8,
        "burning": 7,
        "crushing": 9,
        "squeezing": 8,
    }

    for word, score in severity_map.items():
        if word in text:
            return score

    return 0


import re


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
