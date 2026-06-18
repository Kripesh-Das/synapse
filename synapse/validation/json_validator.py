from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class JSONValidationError(Exception):
    """Raised when LLM output fails validation after retries."""

    def __init__(self, raw_output: str, errors: list[str]):
        self.raw_output = raw_output
        self.errors = errors
        super().__init__(f"JSON validation failed: {'; '.join(errors)}")


def parse_llm_json(
    raw_output: str,
    required_fields: list[str] | None = None,
    max_retries: int = 1,
    reprompt_fn: Any = None,
) -> dict:
    """Parse LLM JSON output with retry and fallback.

    MiniCPM5-1B (1B params) occasionally produces malformed JSON.
    This function:
    1. Tries to parse the raw output
    2. On failure, retries once (if reprompt_fn provided)
    3. On persistent failure, raises JSONValidationError

    Args:
        raw_output: Raw string output from the LLM
        required_fields: Fields that must be present in the parsed JSON
        max_retries: Number of retry attempts (default 1)
        reprompt_fn: Callable that takes (original_output, error_msg) -> str
                     and returns a corrected LLM output

    Returns:
        Parsed JSON dict

    Raises:
        JSONValidationError: If parsing fails after all retries
    """
    errors: list[str] = []

    # Attempt 1: Direct parse
    result = _try_parse(raw_output, required_fields)
    if result is not None:
        return result

    errors.append(f"Initial parse failed for: {raw_output[:200]}")

    # Retry with reprompt
    if reprompt_fn and max_retries > 0:
        try:
            corrected_output = reprompt_fn(raw_output, "Invalid JSON. Respond with valid JSON only.")
            result = _try_parse(corrected_output, required_fields)
            if result is not None:
                return result
            errors.append(f"Retry parse failed for: {corrected_output[:200]}")
        except Exception as e:
            errors.append(f"Reprompt failed: {e}")

    raise JSONValidationError(raw_output, errors)


def _try_parse(raw_output: str, required_fields: list[str] | None = None) -> dict | None:
    """Attempt to parse JSON and validate required fields. Returns None on failure."""
    try:
        # Strip markdown code fences if present
        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines (fences)
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(data, dict):
        return None

    if required_fields:
        missing = [f for f in required_fields if f not in data]
        if missing:
            logger.warning("Missing required fields: %s", missing)
            return None

    return data


def validate_patient_safe(output: dict, patient_message_key: str = "message") -> list[str]:
    """Check that patient-facing output doesn't contain medical advice.

    Returns a list of violation descriptions (empty if clean).
    """
    violations: list[str] = []
    message = output.get(patient_message_key, "")

    if not isinstance(message, str):
        return violations

    # Forbidden patterns — medical advice indicators
    forbidden = [
        r"\byou have\b.*\b(?:disease|pneumonia|infection|condition)\b",
        r"\byou are\b.*\bsick\b",
        r"\bdiagnos(?:is|e|ing)\b",
        r"\bprescri(?:be|ption)\b",
        r"\bmedication\b.*\bshould\b",
        r"\btake\b.*\bpill\b",
        r"\byour condition is\b",
        r"\btriage score\b",
        r"\bscore is\b.*\b\d/5\b",
    ]

    message_lower = message.lower()
    for pattern in forbidden:
        if re.search(pattern, message_lower):
            violations.append(f"Patient message contains medical advice pattern: {pattern}")

    return violations


def validate_triage_score_hidden(output: dict, patient_message_key: str = "message") -> bool:
    """Verify triage_score does NOT appear in patient-facing output.

    Returns True if output is safe (score not leaked).
    """
    message = output.get(patient_message_key, "")
    if not isinstance(message, str):
        return True

    message_lower = message.lower()

    # Check for score leakage patterns
    leakage_patterns = [
        r"\btriage\s+score\b",
        r"\bscore\s*(?:is|:)\s*\d",
        r"\burGENCY\s*(?:level|score)\s*(?:is|:)\s*\d",
        r"\bpriority\s*(?:level|score)\s*(?:is|:)\s*\d",
        r"\bpriority\s+level\b",
    ]

    import re
    for pattern in leakage_patterns:
        if re.search(pattern, message_lower):
            return False

    return True


# Need re for validate_patient_safe
import re
