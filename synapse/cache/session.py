"""Redis session cache for synapse.

Stores active session state with configurable TTL.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from synapse.config import settings

logger = logging.getLogger(__name__)

_redis_client = None


def get_redis():
    """Get or create the Redis client."""
    global _redis_client
    if _redis_client is None:
        import redis
        _redis_client = redis.from_url(settings.redis_url, decode_responses=True)
    return _redis_client


def cache_session(session_id: str, state: dict, ttl: int | None = None) -> None:
    """Cache session state in Redis with TTL."""
    client = get_redis()
    key = f"synapse:session:{session_id}"
    ttl = ttl or settings.session_ttl

    # Serialize state — handle non-JSON-serializable types
    serializable = {}
    for k, v in state.items():
        try:
            json.dumps(v)
            serializable[k] = v
        except (TypeError, ValueError):
            serializable[k] = str(v)

    client.set(key, json.dumps(serializable), ex=ttl)
    logger.debug("Cached session %s (ttl=%ds)", session_id, ttl)


def get_cached_session(session_id: str) -> dict | None:
    """Retrieve cached session state from Redis."""
    client = get_redis()
    key = f"synapse:session:{session_id}"
    data = client.get(key)
    if data:
        return json.loads(data)
    return None


def invalidate_session(session_id: str) -> None:
    """Remove session from cache."""
    client = get_redis()
    key = f"synapse:session:{session_id}"
    client.delete(key)
    logger.debug("Invalidated session %s", session_id)


def extend_session(session_id: str, ttl: int | None = None) -> bool:
    """Extend session TTL. Returns True if key existed."""
    client = get_redis()
    key = f"synapse:session:{session_id}"
    ttl = ttl or settings.session_ttl
    return client.expire(key, ttl)


def session_exists(session_id: str) -> bool:
    """Check if a session is cached."""
    client = get_redis()
    key = f"synapse:session:{session_id}"
    return client.exists(key) > 0


def close_redis() -> None:
    """Close Redis connection."""
    global _redis_client
    if _redis_client is not None:
        _redis_client.close()
        _redis_client = None
