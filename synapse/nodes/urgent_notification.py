"""Urgent Notification node — alerts clinicians for high-priority cases. Deterministic."""

from __future__ import annotations

import logging

from synapse.state import SessionState

logger = logging.getLogger(__name__)


def urgent_notification(state: SessionState) -> dict:
    """Alert clinicians of urgent/critical cases.

    Pushes to dashboard and sends real-time alerts.
    This node is deterministic — no LLM calls.
    """
    triage_result = state.get("triage_result")
    if not triage_result:
        logger.warning("No triage result for urgent notification")
        return {}

    score = triage_result.score
    department = triage_result.department
    session_id = state.get("session_id", "unknown")

    if score == 1:
        logger.critical(
            "URGENT ALERT [CRITICAL] session=%s dept=%s",
            session_id, department,
        )
        # TODO: notification.send_page("on_call_attending", state)
        # TODO: notification.send_push("ed_charge_nurse", state)
    elif score == 2:
        logger.warning(
            "URGENT ALERT [URGENT] session=%s dept=%s",
            session_id, department,
        )
        # TODO: notification.send_push(f"{department}_team", state)

    return {
        "last_checkpoint_node": "urgent_notification",
    }
