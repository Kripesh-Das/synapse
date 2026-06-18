"""Summary Generator node — creates patient message and clinician summary. Hybrid."""

from __future__ import annotations

import json
import logging

from synapse.llm.client import call_llm_json
from synapse.prompts.templates import SUMMARY_PATIENT_PROMPT, get_urgency_level
from synapse.state import SessionState, TriageResult
from synapse.tools.registry import get_tools_for_node
from synapse.validation.json_validator import validate_patient_safe, validate_triage_score_hidden

logger = logging.getLogger(__name__)


def summary_generator(state: SessionState) -> dict:
    """Generate patient-facing message and clinician summary.

    Clinician summary is deterministic (template).
    Patient message uses LLM with limited scope.
    Output is validated for patient safety before returning.
    """
    triage_result: TriageResult = state.get("triage_result") or TriageResult()

    # Clinician summary — deterministic template, no LLM
    clinician_summary = _build_clinician_summary(state, triage_result)

    # Patient message — LLM formats the triage result in patient-friendly language
    prompt = SUMMARY_PATIENT_PROMPT.format(
        triage_score=triage_result.score,
        urgency_level=get_urgency_level(triage_result.score),
        recommended_department=triage_result.department,
        estimated_wait=triage_result.estimated_wait,
        protocol_instructions="Follow standard department routing.",
    )

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Generate a patient-facing message explaining their next steps."},
    ]

    tools = get_tools_for_node("summary_generator")

    try:
        result = call_llm_json(messages, tools=tools)
        patient_message = result.get("message", "")
    except Exception as e:
        logger.error("Summary generator LLM failed: %s", e)
        patient_message = (
            f"Please proceed to the {triage_result.department}. "
            f"Estimated wait time is approximately {triage_result.estimated_wait} minutes. "
            "If your condition worsens, please alert staff immediately."
        )

    # Patient safety validation
    patient_message = _validate_and_sanitize(patient_message, "summary_generator")

    return {
        "patient_facing_message": patient_message,
        "clinician_summary": clinician_summary,
        "last_checkpoint_node": "summary_generator",
    }


def _validate_and_sanitize(message: str, node_name: str) -> str:
    """Validate patient-facing message for safety. Sanitize if violations found."""
    if not message:
        return message

    output = {"message": message}

    # Check for medical advice
    violations = validate_patient_safe(output)
    if violations:
        logger.warning("Patient safety violations in %s: %s", node_name, violations)
        return _safe_fallback()

    # Check for triage score leakage
    if not validate_triage_score_hidden(output):
        logger.warning("Triage score leaked in %s patient message", node_name)
        return _safe_fallback()

    return message


def _safe_fallback() -> str:
    """Return a safe fallback message when validation fails."""
    return "Please proceed to the recommended department. If your condition worsens, please alert staff immediately."


def _build_clinician_summary(state: SessionState, triage_result: TriageResult) -> dict:
    """Build structured clinician summary from extracted data. No LLM."""
    symptoms = state.get("extracted_symptoms", [])
    entities = state.get("extracted_entities", {})

    chief_complaint = symptoms[0].name if symptoms else "Unknown"

    symptom_timeline = "; ".join(
        f"{s.name} ({s.duration}, severity {s.severity}/10)" for s in symptoms
    )

    return {
        "chief_complaint": chief_complaint,
        "history_of_present_illness": symptom_timeline,
        "triage_score": triage_result.score,
        "triage_justification": "; ".join(triage_result.justifications),
        "recommended_department": triage_result.department,
        "estimated_wait": triage_result.estimated_wait,
        "risk_flags": [],
        "ai_confidence": "N/A - deterministic scoring",
        "conversation_turns": state.get("turn_count", 0),
        "emergency_override": state.get("emergency_detected", False),
        "patient_context": entities,
    }
