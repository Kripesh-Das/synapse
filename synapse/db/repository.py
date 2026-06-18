"""Repository functions for persisting synapse data to PostgreSQL."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from synapse.db.connection import get_connection
from synapse.state import Symptom, TriageResult

logger = logging.getLogger(__name__)


def insert_session(
    session_id: str,
    patient_id: str = "",
    source: str = "unknown",
    language: str = "en",
) -> None:
    """Create a new session record."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO sessions (session_id, patient_id, source, language)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (session_id) DO NOTHING
            """,
            (session_id, patient_id, source, language),
        )
        conn.commit()


def insert_conversation_message(
    session_id: str,
    role: str,
    content: str,
) -> None:
    """Insert a single conversation message."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO conversations (session_id, role, content)
            VALUES (%s, %s, %s)
            """,
            (session_id, role, content),
        )
        conn.commit()


def insert_triage_record(
    session_id: str,
    patient_id: str,
    triage_result: TriageResult,
    symptoms: list[Symptom],
    patient_context: dict,
    clinician_summary: dict,
    emergency_detected: bool = False,
    emergency_type: str = "",
) -> None:
    """Insert a triage record after scoring is complete."""
    symptoms_json = json.dumps([
        {"name": s.name, "severity": s.severity, "duration": s.duration, "location": s.location}
        for s in symptoms
    ])

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO triage_records (
                session_id, patient_id, triage_score, department,
                justification, symptoms, patient_context, clinician_summary,
                emergency_detected, emergency_type
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                session_id,
                patient_id,
                triage_result.score,
                triage_result.department,
                "; ".join(triage_result.justifications),
                symptoms_json,
                json.dumps(patient_context),
                json.dumps(clinician_summary),
                emergency_detected,
                emergency_type,
            ),
        )
        conn.commit()


def insert_emergency_incident(
    session_id: str,
    emergency_type: str,
    matched_keyword: str = "",
    raw_message: str = "",
) -> int:
    """Log an emergency incident and return the incident ID."""
    with get_connection() as conn:
        result = conn.execute(
            """
            INSERT INTO emergency_incidents (session_id, emergency_type, matched_keyword, raw_message)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (session_id, emergency_type, matched_keyword, raw_message),
        )
        conn.commit()
        return result.fetchone()[0]


def close_session(session_id: str) -> None:
    """Mark a session as closed."""
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE sessions SET closed_at = NOW(), status = 'closed'
            WHERE session_id = %s
            """,
            (session_id,),
        )
        conn.commit()


def get_session(session_id: str) -> dict | None:
    """Retrieve session metadata."""
    with get_connection() as conn:
        result = conn.execute(
            "SELECT * FROM sessions WHERE session_id = %s",
            (session_id,),
        )
        row = result.fetchone()
        if row:
            return dict(row)
        return None


def get_triage_records(patient_id: str, limit: int = 10) -> list[dict]:
    """Retrieve triage records for a patient."""
    with get_connection() as conn:
        result = conn.execute(
            """
            SELECT * FROM triage_records
            WHERE patient_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (patient_id, limit),
        )
        return [dict(row) for row in result.fetchall()]
