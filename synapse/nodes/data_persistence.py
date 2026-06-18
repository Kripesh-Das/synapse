"""Data Persistence node — saves all interaction data. Deterministic."""

from __future__ import annotations

import logging

from synapse.state import SessionState

logger = logging.getLogger(__name__)


def data_persistence(state: SessionState) -> dict:
    """Save conversation, triage record, and session data.

    This node is deterministic. Gracefully degrades if DB is unavailable.
    """
    session_id = state.get("session_id", "unknown")
    triage_result = state.get("triage_result")

    logger.info(
        "Persisting session %s: score=%s dept=%s turns=%d",
        session_id,
        triage_result.score if triage_result else "N/A",
        triage_result.department if triage_result else "N/A",
        state.get("turn_count", 0),
    )

    _persist_to_postgres(state)
    _persist_to_redis(state)

    return {
        "last_checkpoint_node": "data_persistence",
    }


def _persist_to_postgres(state: SessionState) -> None:
    """Save session, conversation, and triage record to PostgreSQL."""
    try:
        from synapse.db.repository import (
            insert_conversation_message,
            insert_emergency_incident,
            insert_session,
            insert_triage_record,
        )

        session_id = state.get("session_id", "unknown")

        # Create session record
        insert_session(
            session_id=session_id,
            patient_id=state.get("patient_id", ""),
            source=state.get("source", "unknown"),
            language=state.get("language", "en"),
        )

        # Save conversation messages
        messages = state.get("messages", [])
        for msg in messages:
            role = ""
            content = ""
            if hasattr(msg, "type"):
                role = "user" if msg.type == "human" else msg.type
                content = msg.content if hasattr(msg, "content") else ""
            elif isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")

            if role and content:
                insert_conversation_message(session_id, role, content)

        # Save triage record
        triage_result = state.get("triage_result")
        if triage_result:
            insert_triage_record(
                session_id=session_id,
                patient_id=state.get("patient_id", ""),
                triage_result=triage_result,
                symptoms=state.get("extracted_symptoms", []),
                patient_context=state.get("extracted_entities", {}),
                clinician_summary=state.get("clinician_summary", {}),
                emergency_detected=state.get("emergency_detected", False),
                emergency_type=state.get("emergency_type", ""),
            )

        # Log emergency incident if detected
        if state.get("emergency_detected"):
            insert_emergency_incident(
                session_id=session_id,
                emergency_type=state.get("emergency_type", ""),
                raw_message=str(state.get("patient_facing_message", "")),
            )

        logger.info("PostgreSQL persistence complete for session %s", session_id)

    except Exception as e:
        logger.warning("PostgreSQL persistence failed (non-fatal): %s", e)


def _persist_to_redis(state: SessionState) -> None:
    """Cache session state in Redis for fast access."""
    try:
        from synapse.cache.session import cache_session

        session_id = state.get("session_id", "unknown")
        cache_session(session_id, {
            "session_id": session_id,
            "turn_count": state.get("turn_count", 0),
            "sufficient_data": state.get("sufficient_data", False),
            "emergency_detected": state.get("emergency_detected", False),
            "last_checkpoint_node": state.get("last_checkpoint_node", ""),
        })

        logger.info("Redis cache updated for session %s", session_id)

    except Exception as e:
        logger.warning("Redis cache update failed (non-fatal): %s", e)
