"""Session Manager node — initializes session state. Deterministic, no LLM."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from synapse.state import SessionState, TriageResult


def session_manager(state: SessionState) -> dict:
    """Initialize session, load patient record, set safety defaults."""
    now = datetime.now(timezone.utc).isoformat()

    return {
        "session_id": state.get("session_id") or str(uuid.uuid4()),
        "turn_count": 0,
        "max_turns": 20,
        "emergency_detected": False,
        "emergency_type": "",
        "extracted_symptoms": [],
        "extracted_entities": {},
        "triage_result": TriageResult(),
        "retrieved_context": [],
        "patient_facing_message": "",
        "clinician_summary": {},
        "sufficient_data": False,
        "error_message": "",
        "language": state.get("language") or "en",
        "source": state.get("source") or "unknown",
        "timestamp": now,
        "last_checkpoint_node": "session_manager",
        "last_checkpoint_time": now,
    }
