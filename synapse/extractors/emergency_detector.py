from __future__ import annotations

import re
from dataclasses import dataclass

# Keyword categories and their trigger phrases.
# Each phrase is checked case-insensitively against the patient message.
# Order within a category matters — first match wins for that category.
EMERGENCY_KEYWORDS: dict[str, list[str]] = {
    "cardiac": [
        "chest pain",
        "chest tightness",
        "chest feels tight",
        "heart attack",
        "can't breathe",
        "cant breathe",
        "shortness of breath",
        "difficulty breathing",
        "crushing chest pain",
        "pain radiating to arm",
        "pain radiating to jaw",
    ],
    "neurological": [
        "unconscious",
        "seizure",
        "stroke",
        "can't move",
        "cant move",
        "paralyzed",
        "severe headache",
        "worst headache",
        "sudden numbness",
        "face drooping",
        "drooping on one side",
        "arm weakness",
        "speech difficulty",
        "slurred speech",
        "speech is slurred",
    ],
    "trauma": [
        "severe bleeding",
        "bleeding heavily",
        "gunshot",
        "stabbed",
        "broken bone protruding",
        "major accident",
        "car crash",
        "hit my head",
        "head injury",
        "fell from",
        "fall from height",
    ],
    "psychiatric": [
        "suicide",
        "kill myself",
        "want to die",
        "overdose",
        "took pills",
        "self harm",
        "hurting myself",
    ],
    "obstetric": [
        "pregnant and bleeding",
        "labor pains",
        "water broke",
        "pregnant with chest pain",
        "heavy bleeding pregnant",
    ],
    "respiratory": [
        "choking",
        "throat closing",
        "throat is closing",
        "anaphylaxis",
        "severe allergic reaction",
        "tongue swelling",
        "tongue is swelling",
    ],
}


@dataclass
class EmergencyResult:
    detected: bool
    emergency_type: str  # empty string if not detected
    matched_keyword: str  # the keyword that triggered
    confidence: str  # "rule_based_100"

    def to_dict(self) -> dict:
        return {
            "emergency_detected": self.detected,
            "emergency_type": self.emergency_type,
            "matched_keyword": self.matched_keyword,
            "confidence": self.confidence,
        }


def detect_emergency(message: str) -> EmergencyResult:
    """Run keyword matcher against patient message.

    CRITICAL: This function runs BEFORE the LLM sees the message.
    It uses pure keyword matching — no ML, no LLM, no hallucination risk.

    Priority order: cardiac > neurological > trauma > psychiatric > obstetric > respiratory.
    First category with a match wins.
    """
    message_lower = message.lower()

    # Normalize common variations
    message_lower = message_lower.replace("'", "'").replace("'", "'")

    for emergency_type, keywords in EMERGENCY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in message_lower:
                return EmergencyResult(
                    detected=True,
                    emergency_type=emergency_type,
                    matched_keyword=keyword,
                    confidence="rule_based_100",
                )

    return EmergencyResult(
        detected=False,
        emergency_type="",
        matched_keyword="",
        confidence="",
    )
