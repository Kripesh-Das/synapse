"""Session Close node — final message and cleanup. LLM node."""

from __future__ import annotations

import logging

from synapse.llm.client import call_llm_json
from synapse.prompts.templates import SESSION_CLOSE_PROMPT
from synapse.state import SessionState
from synapse.tools.registry import get_tools_for_node
from synapse.validation.json_validator import validate_patient_safe

logger = logging.getLogger(__name__)

SAFE_CLOSING = "Thank you for using synapse. Take care."


def session_close(state: SessionState) -> dict:
    """Send closing message and clean up session.

    Output is validated for patient safety before returning.
    """
    triage_result = state.get("triage_result")
    department = triage_result.department if triage_result else "General Practice"
    language = state.get("language", "en")

    prompt = SESSION_CLOSE_PROMPT.format(
        recommended_department=department,
        language=language,
    )

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Generate a brief closing message for the patient."},
    ]

    tools = get_tools_for_node("session_close")

    try:
        result = call_llm_json(messages, tools=tools)
        closing = result.get("closing_message", SAFE_CLOSING)
    except Exception as e:
        logger.error("Session close LLM failed: %s", e)
        closing = SAFE_CLOSING

    # Patient safety validation
    violations = validate_patient_safe({"message": closing})
    if violations:
        logger.warning("Patient safety violations in session_close: %s", violations)
        closing = SAFE_CLOSING

    return {
        "patient_facing_message": closing,
        "last_checkpoint_node": "session_close",
    }
