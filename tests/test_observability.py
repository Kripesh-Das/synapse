"""Tests for Langfuse observability — uses mocked Langfuse client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from synapse.llm.observability import (
    _NoopTrace,
    is_enabled,
    trace_llm_call,
    trace_node,
)


class TestObservabilityDisabled:
    """Tests when Langfuse is not configured (no keys)."""

    @patch("synapse.llm.observability.settings")
    def test_is_disabled_without_keys(self, mock_settings):
        mock_settings.langfuse_public_key = ""
        mock_settings.langfuse_secret_key = ""
        assert is_enabled() is False

    @patch("synapse.llm.observability.settings")
    def test_trace_llm_call_returns_noop_when_disabled(self, mock_settings):
        mock_settings.langfuse_public_key = ""
        mock_settings.langfuse_secret_key = ""
        trace = trace_llm_call("test", model="test-model")
        assert isinstance(trace, _NoopTrace)

    @patch("synapse.llm.observability.settings")
    def test_trace_node_returns_noop_when_disabled(self, mock_settings):
        mock_settings.langfuse_public_key = ""
        mock_settings.langfuse_secret_key = ""
        trace = trace_node("test_node")
        assert isinstance(trace, _NoopTrace)


class TestNoopTrace:
    def test_noop_end_does_not_raise(self):
        trace = _NoopTrace()
        trace.end(output="test", usage={"tokens": 10})

    def test_noop_context_manager(self):
        with _NoopTrace() as trace:
            trace.end()


class TestObservabilityEnabled:
    """Tests when Langfuse is configured (mocked client)."""

    @patch("synapse.llm.observability.get_langfuse")
    def test_trace_llm_call_creates_generation(self, mock_get_langfuse):
        mock_client = MagicMock()
        mock_trace = MagicMock()
        mock_generation = MagicMock()
        mock_client.trace.return_value = mock_trace
        mock_trace.generation.return_value = mock_generation
        mock_get_langfuse.return_value = mock_client

        trace = trace_llm_call(
            "test_call",
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
        )

        mock_client.trace.assert_called_once()
        mock_trace.generation.assert_called_once()
        trace.end(output={"content": "hi"}, usage={"total_tokens": 10})
        mock_generation.end.assert_called_once()

    @patch("synapse.llm.observability.get_langfuse")
    def test_trace_llm_call_handles_error(self, mock_get_langfuse):
        mock_client = MagicMock()
        mock_trace = MagicMock()
        mock_generation = MagicMock()
        mock_client.trace.return_value = mock_trace
        mock_trace.generation.return_value = mock_generation
        mock_get_langfuse.return_value = mock_client

        trace = trace_llm_call("test_call")
        trace.end(error="something went wrong")
        mock_generation.end.assert_called_once()

    @patch("synapse.llm.observability.get_langfuse")
    def test_trace_node_creates_trace(self, mock_get_langfuse):
        mock_client = MagicMock()
        mock_trace = MagicMock()
        mock_client.trace.return_value = mock_trace
        mock_get_langfuse.return_value = mock_client

        trace = trace_node("symptom_collector", session_id="sess-123")
        mock_client.trace.assert_called_once()
        trace.end()
