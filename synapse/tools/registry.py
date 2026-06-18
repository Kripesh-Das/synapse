"""Tool registry — maps tool names to their implementations.

The harness uses this to dispatch LLM tool calls to the right function.
"""

from __future__ import annotations

from typing import Any, Callable

from synapse.tools.conversation import (
    end_conversation,
    request_human,
    switch_language,
)
from synapse.tools.knowledge import (
    check_wait_times,
    get_department_info,
    search_hospital_kb,
)
from synapse.tools.symptoms import (
    ask_symptom_details,
    check_triage_ready,
    confirm_symptom,
    reject_symptom,
    summarize_symptoms,
    trigger_emergency_alert,
    update_patient_context,
)

TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "end_conversation": end_conversation,
    "request_human": request_human,
    "switch_language": switch_language,
    "ask_symptom_details": ask_symptom_details,
    "confirm_symptom": confirm_symptom,
    "reject_symptom": reject_symptom,
    "summarize_symptoms": summarize_symptoms,
    "update_patient_context": update_patient_context,
    "check_triage_ready": check_triage_ready,
    "trigger_emergency_alert": trigger_emergency_alert,
    "search_hospital_kb": search_hospital_kb,
    "get_department_info": get_department_info,
    "check_wait_times": check_wait_times,
}


def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool by name with the given arguments.

    Returns the tool result dict, or an error dict if the tool is not found.
    """
    tool_fn = TOOL_REGISTRY.get(name)
    if tool_fn is None:
        return {"error": f"Tool '{name}' not found"}

    try:
        return tool_fn(**arguments)
    except TypeError as e:
        return {"error": f"Invalid parameters: {e}"}
    except Exception as e:
        return {"error": f"Tool execution failed: {e}"}


# Progressive tool disclosure — which tools are available at each node
NODE_TOOLS: dict[str, list[str]] = {
    "welcome_agent": ["end_conversation", "request_human", "switch_language"],
    "symptom_collector": [
        "end_conversation", "request_human", "switch_language",
        "ask_symptom_details", "confirm_symptom", "reject_symptom",
        "summarize_symptoms", "update_patient_context", "check_triage_ready",
        "trigger_emergency_alert",
    ],
    "summary_generator": [
        "end_conversation", "request_human", "switch_language",
        "search_hospital_kb", "get_department_info", "check_wait_times",
    ],
    "session_close": ["end_conversation", "request_human", "switch_language"],
}


def get_tools_for_node(node_name: str) -> list[dict]:
    """Get OpenAI-format tool definitions for a specific node."""
    from synapse.tools.conversation import CONVERSATION_TOOLS
    from synapse.tools.knowledge import KNOWLEDGE_TOOLS
    from synapse.tools.symptoms import SYMPTOM_TOOLS

    all_tools = CONVERSATION_TOOLS + SYMPTOM_TOOLS + KNOWLEDGE_TOOLS
    allowed = set(NODE_TOOLS.get(node_name, []))

    return [t for t in all_tools if t["function"]["name"] in allowed]
