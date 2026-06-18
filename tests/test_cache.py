"""Tests for Redis session cache — uses mocked Redis client."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from synapse.cache.session import (
    cache_session,
    extend_session,
    get_cached_session,
    invalidate_session,
    session_exists,
)


class TestSessionCache:
    @patch("synapse.cache.session.get_redis")
    def test_cache_session(self, mock_get_redis):
        mock_client = MagicMock()
        mock_get_redis.return_value = mock_client

        cache_session("sess-123", {"turn_count": 5, "sufficient_data": False})

        mock_client.set.assert_called_once()
        call_args = mock_client.set.call_args
        assert "synapse:session:sess-123" == call_args[0][0]
        assert call_args[1].get("ex", call_args[0][2] if len(call_args[0]) > 2 else 0) > 0  # TTL > 0

    @patch("synapse.cache.session.get_redis")
    def test_get_cached_session_found(self, mock_get_redis):
        mock_client = MagicMock()
        mock_client.get.return_value = json.dumps({"turn_count": 5})
        mock_get_redis.return_value = mock_client

        result = get_cached_session("sess-123")
        assert result is not None
        assert result["turn_count"] == 5

    @patch("synapse.cache.session.get_redis")
    def test_get_cached_session_not_found(self, mock_get_redis):
        mock_client = MagicMock()
        mock_client.get.return_value = None
        mock_get_redis.return_value = mock_client

        result = get_cached_session("nonexistent")
        assert result is None

    @patch("synapse.cache.session.get_redis")
    def test_invalidate_session(self, mock_get_redis):
        mock_client = MagicMock()
        mock_get_redis.return_value = mock_client

        invalidate_session("sess-123")

        mock_client.delete.assert_called_once_with("synapse:session:sess-123")

    @patch("synapse.cache.session.get_redis")
    def test_extend_session(self, mock_get_redis):
        mock_client = MagicMock()
        mock_client.expire.return_value = True
        mock_get_redis.return_value = mock_client

        result = extend_session("sess-123")
        assert result is True
        mock_client.expire.assert_called_once()

    @patch("synapse.cache.session.get_redis")
    def test_session_exists_true(self, mock_get_redis):
        mock_client = MagicMock()
        mock_client.exists.return_value = 1
        mock_get_redis.return_value = mock_client

        assert session_exists("sess-123") is True

    @patch("synapse.cache.session.get_redis")
    def test_session_exists_false(self, mock_get_redis):
        mock_client = MagicMock()
        mock_client.exists.return_value = 0
        mock_get_redis.return_value = mock_client

        assert session_exists("nonexistent") is False

    @patch("synapse.cache.session.get_redis")
    def test_cache_session_handles_non_serializable(self, mock_get_redis):
        """Non-JSON-serializable values should be converted to strings."""
        mock_client = MagicMock()
        mock_get_redis.return_value = mock_client

        # TriageResult is a dataclass, not JSON-serializable by default
        from synapse.state import TriageResult
        cache_session("sess-123", {"triage": TriageResult(score=2)})

        # Should not raise
        mock_client.set.assert_called_once()
