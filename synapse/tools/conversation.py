"""Conversation control tools — available at all LLM nodes."""

from __future__ import annotations

from typing import Any


def end_conversation(reason: str, final_message: str = "") -> dict[str, Any]:
    """End the current triage session gracefully."""
    return {
        "status": "ended",
        "reason": reason,
        "message_shown": bool(final_message),
    }


def request_human(reason: str, urgency: str, notes: str = "") -> dict[str, Any]:
    """Request a human staff member to take over the conversation."""
    return {
        "status": "queued",
        "reason": reason,
        "urgency": urgency,
        "notes": notes,
    }


def switch_language(language_code: str, confirm_with_patient: bool = True) -> dict[str, Any]:
    """Switch the conversation language."""
    return {
        "status": "switched",
        "language": language_code,
        "confirmed": confirm_with_patient,
    }


# Tool definitions for the LLM (OpenAI function calling format)
CONVERSATION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "end_conversation",
            "description": "End the current triage session. Use when patient wants to stop, is abusive, or for technical errors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "enum": ["patient_request", "abusive", "technical_error", "max_length", "other"],
                        "description": "Why the conversation is ending",
                    },
                    "final_message": {
                        "type": "string",
                        "description": "Optional closing message to show patient",
                    },
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_human",
            "description": "Request a human staff member. Use when patient explicitly asks, is frustrated, or situation is complex.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "enum": ["patient_request", "frustrated", "complex_case", "language_barrier", "emotional_support", "other"],
                        "description": "Why human handoff is needed",
                    },
                    "urgency": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "How urgently staff should respond",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Context for the human staff member",
                    },
                },
                "required": ["reason", "urgency"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "switch_language",
            "description": "Switch the conversation language when patient indicates preference.",
            "parameters": {
                "type": "object",
                "properties": {
                    "language_code": {
                        "type": "string",
                        "enum": ["en", "es", "fr", "zh", "hi", "ar", "other"],
                        "description": "ISO language code",
                    },
                    "confirm_with_patient": {
                        "type": "boolean",
                        "description": "Whether to ask patient to confirm",
                        "default": True,
                    },
                },
                "required": ["language_code"],
            },
        },
    },
]
