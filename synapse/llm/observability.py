"""Langfuse observability integration for synapse.

Provides tracing for LLM calls, tool calls, and node execution.
Gracefully degrades if Langfuse is not configured.
"""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Any, Callable

from synapse.config import settings

logger = logging.getLogger(__name__)

_langfuse_client = None
_enabled = False


def get_langfuse():
    """Get or create the Langfuse client. Returns None if not configured."""
    global _langfuse_client, _enabled

    if _langfuse_client is not None:
        return _langfuse_client

    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.debug("Langfuse not configured — tracing disabled")
        _enabled = False
        return None

    try:
        from langfuse import Langfuse
        _langfuse_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        _enabled = True
        logger.info("Langfuse tracing enabled")
        return _langfuse_client
    except Exception as e:
        logger.warning("Failed to initialize Langfuse: %s", e)
        _enabled = False
        return None


def is_enabled() -> bool:
    """Check if Langfuse tracing is enabled."""
    return _enabled


def trace_llm_call(
    name: str,
    model: str = "",
    messages: list[dict] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Create a Langfuse trace for an LLM call.

    Returns a context manager-like object with .end() method.
    """
    client = get_langfuse()
    if client is None:
        return _NoopTrace()

    try:
        trace = client.trace(name=name, **kwargs)
        generation = trace.generation(
            name=name,
            model=model,
            input=messages,
            **{k: v for k, v in kwargs.items() if k not in ("metadata", "tags")},
        )
        return _LangfuseTrace(trace=trace, generation=generation, client=client)
    except Exception as e:
        logger.warning("Langfuse trace creation failed: %s", e)
        return _NoopTrace()


def trace_node(node_name: str, session_id: str = "") -> dict[str, Any]:
    """Create a Langfuse trace for a graph node execution."""
    client = get_langfuse()
    if client is None:
        return _NoopTrace()

    try:
        trace = client.trace(
            name=f"node:{node_name}",
            session_id=session_id,
            metadata={"node": node_name},
        )
        return _LangfuseTrace(trace=trace, generation=None, client=client)
    except Exception as e:
        logger.warning("Langfuse trace creation failed: %s", e)
        return _NoopTrace()


class _LangfuseTrace:
    """Wrapper around Langfuse trace with end() method."""

    def __init__(self, trace, generation, client):
        self._trace = trace
        self._generation = generation
        self._client = client
        self._start_time = time.time()

    def end(self, output: Any = None, usage: dict | None = None, error: str | None = None):
        """End the trace/generation."""
        try:
            if self._generation:
                kwargs = {}
                if output is not None:
                    kwargs["output"] = output
                if usage:
                    kwargs["usage"] = usage
                if error:
                    kwargs["level"] = "ERROR"
                    kwargs["status_message"] = error
                self._generation.end(**kwargs)

            if self._trace:
                if error:
                    self._trace.update(level="ERROR", status_message=error)
                self._trace.update(metadata={
                    "duration_ms": int((time.time() - self._start_time) * 1000),
                })
        except Exception as e:
            logger.debug("Langfuse end failed: %s", e)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.end()


class _NoopTrace:
    """No-op trace when Langfuse is disabled."""

    def end(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
