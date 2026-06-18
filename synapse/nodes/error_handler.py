"""Error Handler node — graceful degradation when systems fail. Deterministic."""

from __future__ import annotations

import logging

from synapse.state import SessionState

logger = logging.getLogger(__name__)

FALLBACK_MESSAGES = {
    "en": "We apologize, but we're experiencing a technical issue. Please proceed to the reception desk where our staff will assist you. Your information has been saved.",
    "es": "Lo sentimos, estamos experimentando un problema técnico. Por favor, diríjase a la recepción donde nuestro personal le asistirá. Su información ha sido guardada.",
}


def error_handler(state: SessionState) -> dict:
    """Handle errors with graceful degradation.

    This node is deterministic. No LLM calls.
    """
    session_id = state.get("session_id", "unknown")
    error = state.get("error_message", "Unknown error")

    logger.error("Error handler triggered for session %s: %s", session_id, error)

    language = state.get("language", "en")
    fallback = FALLBACK_MESSAGES.get(language, FALLBACK_MESSAGES["en"])

    # TODO: helpdesk.create_ticket(type="ai_system_failure", ...)
    # TODO: if critical: pagerduty.alert("ai-system-oncall", error)

    return {
        "patient_facing_message": fallback,
        "last_checkpoint_node": "error_handler",
    }
