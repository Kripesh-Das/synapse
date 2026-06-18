#!/usr/bin/env python3
"""CLI runner for synapse triage agent.

Usage:
    # With mock LLM (no SGLang needed):
    python -m synapse.cli --mock

    # With real SGLang server:
    python -m synapse.cli

    # Single message mode:
    python -m synapse.cli --mock --message "I have chest pain"
"""

from __future__ import annotations

import argparse
import json
import sys

from synapse.graph import build_graph
from synapse.llm.client import set_llm_override
from synapse.state import SessionState


def mock_llm(messages: list[dict], tools: list[dict] | None = None) -> dict:
    """Deterministic mock LLM for testing without SGLang."""
    system_msg = ""
    for m in messages:
        if m.get("role") == "system":
            system_msg = m.get("content", "")
            break

    sys_lower = system_msg.lower()

    if "welcome" in sys_lower or "greet" in sys_lower:
        return {
            "content": json.dumps({
                "message": "Welcome to synapse. I'm here to help triage your symptoms. What brings you in today?",
                "next_action": "ask_symptoms",
            })
        }
    elif "collecting symptoms" in sys_lower or "symptom" in sys_lower:
        return {
            "content": json.dumps({
                "message": "Can you tell me more about that? How long have you had this symptom, and how would you rate the severity from 1-10?",
                "sufficient_data": False,
                "missing_fields": ["duration", "severity"],
            })
        }
    elif "explaining the next steps" in sys_lower or "summary" in sys_lower:
        return {
            "content": json.dumps({
                "message": "Based on your symptoms, please proceed to the appropriate department. If your condition worsens, please alert staff immediately.",
                "include_map": True,
                "include_preparation": True,
            })
        }
    elif "closing message" in sys_lower or "goodbye" in sys_lower:
        return {
            "content": json.dumps({
                "closing_message": "Thank you for using synapse. Take care and we hope you feel better soon.",
                "feedback_request": "How was your experience? (1-5)",
            })
        }

    return {"content": json.dumps({"message": "I'm here to help. Please describe your symptoms."})}


def run_interactive(use_mock: bool = False):
    """Run interactive triage session."""
    if use_mock:
        set_llm_override(mock_llm)
        print("[Mock LLM mode — no SGLang required]\n")

    graph = build_graph()

    print("=" * 60)
    print("  synapse Hospital AI Triage Assistant")
    print("  Type 'quit' to exit, 'emergency' to test emergency flow")
    print("=" * 60)
    print()

    messages = []
    state: SessionState = {
        "messages": messages,
        "extracted_symptoms": [],
        "source": "cli",
        "language": "en",
    }

    try:
        result = graph.invoke(state)
    except Exception as e:
        print(f"\nError: {e}")
        set_llm_override(None)
        return

    print(f"Agent: {result.get('patient_facing_message', '')}\n")

    # Interactive loop (simplified — single-turn for now)
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        if user_input.lower() == "emergency":
            user_input = "I have severe chest pain and can't breathe"

        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            result = graph.invoke({
                "messages": list(messages),
                "extracted_symptoms": result.get("extracted_symptoms", []),
                "source": "cli",
                "language": "en",
                "emergency_detected": result.get("emergency_detected", False),
                "emergency_type": result.get("emergency_type", ""),
                "turn_count": result.get("turn_count", 0) + 1,
                "sufficient_data": result.get("sufficient_data", False),
            })
        except Exception as e:
            print(f"\nError: {e}")
            break

        print(f"\nAgent: {result.get('patient_facing_message', '')}\n")

        if result.get("last_checkpoint_node") == "session_close":
            break

        # Show triage info if available
        triage = result.get("triage_result")
        if triage and hasattr(triage, "score") and triage.score < 5:
            print(f"  [Triage: score={triage.score}, dept={triage.department}]")

    set_llm_override(None)


def run_single(message: str, use_mock: bool = False):
    """Run a single message through the graph."""
    if use_mock:
        set_llm_override(mock_llm)

    graph = build_graph()

    state: SessionState = {
        "messages": [{"role": "user", "content": message}],
        "extracted_symptoms": [],
        "source": "cli",
        "language": "en",
    }

    result = graph.invoke(state)

    print(json.dumps({
        "message": result.get("patient_facing_message", ""),
        "emergency_detected": result.get("emergency_detected", False),
        "emergency_type": result.get("emergency_type", ""),
        "triage_score": result.get("triage_result", {}).score if result.get("triage_result") else None,
        "department": result.get("triage_result", {}).department if result.get("triage_result") else None,
        "last_node": result.get("last_checkpoint_node", ""),
    }, indent=2))

    set_llm_override(None)


def main():
    parser = argparse.ArgumentParser(description="synapse Triage Agent CLI")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM (no SGLang needed)")
    parser.add_argument("--message", "-m", type=str, help="Single message mode")
    args = parser.parse_args()

    if args.message:
        run_single(args.message, use_mock=args.mock)
    else:
        run_interactive(use_mock=args.mock)


if __name__ == "__main__":
    main()
