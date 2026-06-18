"""PostgreSQL connection manager for synapse.

Provides connection pooling and table setup via psycopg3.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator

from synapse.config import settings

logger = logging.getLogger(__name__)

_pool = None


def get_pool():
    """Get or create the connection pool."""
    global _pool
    if _pool is None:
        from psycopg_pool import ConnectionPool
        _pool = ConnectionPool(
            settings.database_url,
            min_size=2,
            max_size=10,
            open=True,
        )
    return _pool


def create_tables() -> None:
    """Create database tables if they don't exist."""
    from synapse.db.schema import get_schema_sql
    pool = get_pool()
    with pool.connection() as conn:
        conn.execute(get_schema_sql())
        conn.commit()
    logger.info("Database tables created/verified")


@contextmanager
def get_connection() -> Generator:
    """Get a database connection from the pool."""
    pool = get_pool()
    with pool.connection() as conn:
        yield conn


def close_pool() -> None:
    """Close the connection pool."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
