"""Symptom collection tools — available at symptom_collector node."""

from __future__ import annotations

from typing import Any

from synapse.state import Symptom


def ask_symptom_details(symptom_name: str, aspect: str, context_hint: str = "") -> dict[str, Any]:
    """Ask the patient for specific details about a symptom."""
    return {
        "symptom_name": symptom_name,
        "aspect": aspect,
        "context_hint": context_hint,
        "status": "awaiting_patient_response",
    }


def confirm_symptom(symptom_name: str, details: dict) -> dict[str, Any]:
    """Confirm extracted symptom details with patient."""
    return {
        "symptom_name": symptom_name,
        "details": details,
        "status": "awaiting_confirmation",
    }


def reject_symptom(symptom_name: str, reason: str) -> dict[str, Any]:
    """Remove an incorrectly extracted symptom."""
    return {
        "symptom_name": symptom_name,
        "reason": reason,
        "status": "removed",
    }


def summarize_symptoms(format: str = "brief") -> dict[str, Any]:
    """Get a summary of all symptoms collected so far."""
    return {
        "format": format,
        "status": "summary_generated",
    }


def update_patient_context(field: str, value: str, confidence: float = 0.8) -> dict[str, Any]:
    """Update patient context with new information."""
    return {
        "field": field,
        "value": value,
        "confidence": confidence,
        "status": "updated",
    }


def check_triage_ready(min_symptoms: int = 1) -> dict[str, Any]:
    """Check if enough information has been collected for triage."""
    return {
        "min_symptoms": min_symptoms,
        "status": "checked",
    }


def trigger_emergency_alert(emergency_type: str, patient_location: str, details: str = "") -> dict[str, Any]:
    """Trigger emergency alert to hospital staff. ONLY for life-threatening situations."""
    return {
        "alert_id": f"ALERT-{hash(patient_location) % 10000:04d}",
        "status": "dispatched",
        "emergency_type": emergency_type,
        "patient_location": patient_location,
    }


SYMPTOM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ask_symptom_details",
            "description": "Ask the patient for specific details about a symptom. Primary tool for symptom collection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symptom_name": {"type": "string", "description": "The symptom to ask about"},
                    "aspect": {
                        "type": "string",
                        "enum": ["duration", "severity", "location", "quality", "triggers", "relieving_factors", "associated_symptoms", "onset"],
                        "description": "What specific detail to ask about",
                    },
                    "context_hint": {"type": "string", "description": "Optional hint about why this detail matters"},
                },
                "required": ["symptom_name", "aspect"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_symptom",
            "description": "Confirm that you correctly understood a symptom the patient described.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symptom_name": {"type": "string", "description": "The symptom to confirm"},
                    "details": {
                        "type": "object",
                        "properties": {
                            "duration": {"type": "string"},
                            "severity": {"type": "integer", "minimum": 1, "maximum": 10},
                            "location": {"type": "string"},
                            "quality": {"type": "string"},
                        },
                        "description": "The extracted details to confirm",
                    },
                },
                "required": ["symptom_name", "details"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reject_symptom",
            "description": "Remove a symptom from the collected list if patient says they don't have it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symptom_name": {"type": "string", "description": "The symptom to remove"},
                    "reason": {
                        "type": "string",
                        "enum": ["patient_denied", "extraction_error", "duplicate", "other"],
                        "description": "Why the symptom is being rejected",
                    },
                },
                "required": ["symptom_name", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_symptoms",
            "description": "Get a summary of all symptoms collected so far.",
            "parameters": {
                "type": "object",
                "properties": {
                    "format": {"type": "string", "enum": ["brief", "detailed", "structured"], "default": "brief"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_patient_context",
            "description": "Update patient context with demographics, allergies, or medications.",
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "enum": ["age", "gender", "pregnancy_status", "allergies", "medications", "medical_history", "primary_concern", "language_preference"],
                        "description": "Which field to update",
                    },
                    "value": {"type": "string", "description": "The value to store"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.8},
                },
                "required": ["field", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_triage_ready",
            "description": "Check if enough information has been collected to run triage scoring.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_symptoms": {"type": "integer", "default": 1, "minimum": 1},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trigger_emergency_alert",
            "description": "TRIGGER EMERGENCY ALERT. ONLY for life-threatening symptoms: chest pain with SOB, severe bleeding, loss of consciousness, suicidal intent, stroke signs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "emergency_type": {
                        "type": "string",
                        "enum": ["cardiac", "respiratory", "trauma", "neurological", "psychiatric", "obstetric", "allergic_reaction", "other_life_threatening"],
                        "description": "Type of emergency",
                    },
                    "patient_location": {"type": "string", "description": "Where the patient is currently located"},
                    "details": {"type": "string", "description": "Brief description of what patient reported"},
                },
                "required": ["emergency_type", "patient_location"],
            },
        },
    },
]
