from __future__ import annotations

import logging
from typing import Any, Callable

from openai import OpenAI

from synapse.config import settings

logger = logging.getLogger(__name__)

_client: OpenAI | None = None
_mock_fn: Callable[[list[dict], list[dict] | None], dict[str, Any]] | None = None


def set_llm_override(fn: Callable[[list[dict], list[dict] | None], dict[str, Any]] | None) -> None:
    """Inject a mock LLM function for testing. Pass None to restore real client."""
    global _mock_fn
    _mock_fn = fn


def get_llm_client() -> OpenAI:
    """Get or create the OpenAI-compatible client pointing at SGLang."""
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )
    return _client


def call_llm(
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    trace_name: str = "llm_call",
) -> dict[str, Any]:
    """Call the LLM and return the response.

    Returns:
        {"content": str|None, "tool_calls": list[dict]|None, "usage": dict}
    """
    if _mock_fn is not None:
        return _mock_fn(messages, tools)

    from synapse.llm.observability import trace_llm_call

    client = get_llm_client()

    kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": temperature or settings.llm_temperature,
        "max_tokens": max_tokens or settings.llm_max_tokens,
    }

    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    trace = trace_llm_call(
        name=trace_name,
        model=settings.llm_model,
        messages=messages,
    )

    try:
        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message

        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        result: dict[str, Any] = {
            "content": message.content,
            "tool_calls": None,
            "usage": usage,
        }

        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in message.tool_calls
            ]

        trace.end(output=result, usage=usage)
        return result

    except Exception as e:
        trace.end(error=str(e))
        logger.error("LLM call failed: %s", e)
        raise


def call_llm_json(
    messages: list[dict],
    required_fields: list[str] | None = None,
    tools: list[dict] | None = None,
    max_retries: int = 1,
    trace_name: str = "llm_json",
) -> dict:
    """Call LLM and parse response as JSON with retry.

    Strips markdown fences, validates required fields, retries once on failure.
    """
    from synapse.validation.json_validator import JSONValidationError, parse_llm_json

    response = call_llm(messages, tools=tools, trace_name=trace_name)
    content = response.get("content") or ""

    return parse_llm_json(
        content,
        required_fields=required_fields,
        max_retries=max_retries,
        reprompt_fn=lambda orig, err: _reprompt_for_json(messages, err),
    )


def _reprompt_for_json(original_messages: list[dict], error: str) -> str:
    """Send a follow-up message asking the LLM to fix its JSON output."""
    followup = original_messages + [
        {
            "role": "user",
            "content": f"Your response was not valid JSON. {error}\n\n"
            "Please respond again with ONLY a valid JSON object, no other text.",
        }
    ]
    response = call_llm(followup, trace_name="llm_json_retry")
    return response.get("content") or ""
