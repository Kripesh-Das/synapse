"""Welcome Agent node — greets patient, explains system. LLM node."""

from __future__ import annotations

import json
import logging

from synapse.llm.client import call_llm_json
from synapse.prompts.templates import WELCOME_SYSTEM_PROMPT
from synapse.state import SessionState
from synapse.tools.registry import get_tools_for_node

logger = logging.getLogger(__name__)


def welcome_agent(state: SessionState) -> dict:
    """Greet patient, explain system, collect initial concern."""
    messages = [
        {"role": "system", "content": WELCOME_SYSTEM_PROMPT},
        {"role": "user", "content": "New patient session started. Greet them and ask what brings them in."},
    ]

    tools = get_tools_for_node("welcome_agent")

    try:
        result = call_llm_json(messages, tools=tools)
        return {
            "patient_facing_message": result.get("message", "Welcome. How can I help you today?"),
            "last_checkpoint_node": "welcome_agent",
        }
    except Exception as e:
        logger.error("Welcome agent LLM failed: %s", e)
        return {
            "patient_facing_message": "Welcome to synapse. How can I help you today?",
            "last_checkpoint_node": "welcome_agent",
        }
