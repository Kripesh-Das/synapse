from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from langgraph.graph import add_messages
from typing_extensions import Annotated, TypedDict


@dataclass
class Symptom:
    name: str
    duration: str = ""
    severity: int = 0  # 1-10
    location: str = ""
    quality: str = ""
    triggers: str = ""
    relieving_factors: str = ""
    associated_symptoms: list[str] = field(default_factory=list)
    onset: str = ""


@dataclass
class TriageResult:
    score: int = 5  # 1=critical, 5=low
    justifications: list[str] = field(default_factory=list)
    department: str = ""
    estimated_wait: int = 0  # minutes


class SessionState(TypedDict, total=False):
    # Session
    session_id: str
    patient_id: str
    turn_count: int
    max_turns: int

    # Conversation (uses add_messages reducer — append, don't overwrite)
    messages: Annotated[list, add_messages]

    # Extracted data (populated by deterministic extractors)
    extracted_symptoms: list[Symptom]
    extracted_entities: dict  # medications, allergies, prior_conditions

    # Safety flags (set by rules engine, NEVER by LLM)
    emergency_detected: bool
    emergency_type: str  # cardiac, neurological, trauma, psychiatric, obstetric

    # Triage (computed by clinical rules engine)
    triage_result: TriageResult

    # RAG
    retrieved_context: list[dict]

    # Output
    patient_facing_message: str
    clinician_summary: dict

    # Checkpointing
    last_checkpoint_node: str
    last_checkpoint_time: str  # ISO format

    # Metadata
    source: str  # kiosk, web, mobile, reception
    language: str
    timestamp: str
    sufficient_data: bool
    error_message: str
