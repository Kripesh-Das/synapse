"""synapse Triage Graph — wires all nodes into a LangGraph StateGraph.

Entry point: build_graph()
"""

from __future__ import annotations

import logging
import logging
from typing import Any, Callable

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from synapse.state import SessionState

logger = logging.getLogger(__name__)


def _route_after_symptom_collector(state: SessionState) -> str:
    """Decide next node after symptom_collector based on state."""
    if state.get("emergency_detected"):
        return "emergency_handler"

    if state.get("sufficient_data"):
        return "triage_engine"

    turn_count = state.get("turn_count", 0)
    max_turns = state.get("max_turns", 20)
    if turn_count >= max_turns:
        return "triage_engine"

    return "symptom_collector"


def _route_after_triage(state: SessionState) -> str:
    """Decide next node after triage_engine based on score."""
    triage_result = state.get("triage_result")
    if triage_result and triage_result.score <= 2:
        return "urgent_notification"
    return "summary_generator"


def _safe_node(fn: Callable, node_name: str) -> Callable:
    """Wrap a node function with error handling.

    If the node raises an exception, sets error_message in state
    so the graph routes to error_handler.
    """
    def wrapper(state: SessionState) -> dict:
        try:
            return fn(state)
        except Exception as e:
            logger.error("Node %s failed: %s", node_name, e)
            return {
                "error_message": f"{node_name} failed: {e}",
                "last_checkpoint_node": node_name,
            }
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


def build_graph(checkpointer=None) -> StateGraph:
    """Construct and compile the synapse triage graph.

    Args:
        checkpointer: LangGraph checkpointer for state persistence.
                     If None, auto-detects PostgresSaver or falls back to MemorySaver.

    Returns:
        Compiled StateGraph ready for invocation.
    """
    from synapse.nodes.data_persistence import data_persistence
    from synapse.nodes.emergency_handler import emergency_handler
    from synapse.nodes.error_handler import error_handler
    from synapse.nodes.session_close import session_close
    from synapse.nodes.session_manager import session_manager
    from synapse.nodes.summary_generator import summary_generator
    from synapse.nodes.symptom_collector import symptom_collector
    from synapse.nodes.triage_engine import triage_engine
    from synapse.nodes.urgent_notification import urgent_notification
    from synapse.nodes.welcome_agent import welcome_agent

    # Auto-detect checkpointer if not provided
    if checkpointer is None:
        checkpointer = _get_default_checkpointer()

    graph = StateGraph(SessionState)

    # Add all nodes with error handling wrappers
    graph.add_node("session_manager", _safe_node(session_manager, "session_manager"))
    graph.add_node("welcome_agent", _safe_node(welcome_agent, "welcome_agent"))
    graph.add_node("symptom_collector", _safe_node(symptom_collector, "symptom_collector"))
    graph.add_node("emergency_handler", _safe_node(emergency_handler, "emergency_handler"))
    graph.add_node("triage_engine", _safe_node(triage_engine, "triage_engine"))
    graph.add_node("urgent_notification", _safe_node(urgent_notification, "urgent_notification"))
    graph.add_node("summary_generator", _safe_node(summary_generator, "summary_generator"))
    graph.add_node("data_persistence", _safe_node(data_persistence, "data_persistence"))
    graph.add_node("session_close", _safe_node(session_close, "session_close"))
    graph.add_node("error_handler", _safe_node(error_handler, "error_handler"))

    # Entry point
    graph.set_entry_point("session_manager")

    # Linear edges
    graph.add_edge("session_manager", "welcome_agent")
    graph.add_edge("welcome_agent", "symptom_collector")
    graph.add_edge("emergency_handler", "session_close")
    graph.add_edge("urgent_notification", "summary_generator")
    graph.add_edge("summary_generator", "data_persistence")
    graph.add_edge("session_close", END)

    # Error handler always goes to session_close
    graph.add_edge("error_handler", "session_close")

    # Conditional edges
    graph.add_conditional_edges(
        "symptom_collector",
        _route_after_symptom_collector,
        {
            "emergency_handler": "emergency_handler",
            "triage_engine": "triage_engine",
            "symptom_collector": "symptom_collector",
        },
    )

    graph.add_conditional_edges(
        "triage_engine",
        _route_after_triage,
        {
            "urgent_notification": "urgent_notification",
            "summary_generator": "summary_generator",
        },
    )

    # data_persistence can fail → error_handler
    graph.add_conditional_edges(
        "data_persistence",
        lambda state: "error_handler" if state.get("error_message") else "session_close",
        {
            "error_handler": "error_handler",
            "session_close": "session_close",
        },
    )

    return graph.compile(checkpointer=checkpointer)


def _get_default_checkpointer():
    """Try PostgresSaver, fall back to MemorySaver."""
    try:
        from synapse.config import settings
        if settings.database_url:
            from langgraph.checkpoint.postgres import PostgresSaver
            with PostgresSaver.from_conn_string(settings.database_url) as saver:
                saver.setup()
            # Re-open for persistent use
            return PostgresSaver.from_conn_string(settings.database_url)
    except Exception as e:
        logger.info("PostgresSaver unavailable (%s), using MemorySaver", e)

    return MemorySaver()
