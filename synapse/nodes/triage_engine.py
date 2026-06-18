"""Triage Engine node — computes score and department. Deterministic rules engine."""

from __future__ import annotations

import logging

from synapse.state import SessionState
from synapse.triage.rules_engine import TriageRulesEngine

logger = logging.getLogger(__name__)


def triage_engine(state: SessionState) -> dict:
    """Compute triage score and recommended department.

    This node is fully deterministic. No LLM calls.
    """
    symptoms = state.get("extracted_symptoms", [])
    entities = state.get("extracted_entities", {})

    # Extract age and pregnancy status from entities
    age = entities.get("age", 0)
    if isinstance(age, str):
        try:
            age = int(age)
        except ValueError:
            age = 0

    pregnancy = entities.get("pregnancy_status", False)
    if isinstance(pregnancy, str):
        pregnancy = pregnancy.lower() in ("true", "yes", "pregnant")

    engine = TriageRulesEngine()
    result = engine.compute_triage(
        symptoms=symptoms,
        age=age,
        pregnancy_status=pregnancy,
    )

    logger.info(
        "Triage computed: score=%d dept=%s justifications=%s",
        result.score, result.department, result.justifications,
    )

    return {
        "triage_result": result,
        "last_checkpoint_node": "triage_engine",
    }
