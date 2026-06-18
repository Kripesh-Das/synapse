"""PostgreSQL schema for synapse triage records.

Run create_tables() once to initialize the database.
"""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    patient_id      TEXT,
    source          TEXT NOT NULL DEFAULT 'unknown',
    language        TEXT NOT NULL DEFAULT 'en',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at       TIMESTAMPTZ,
    status          TEXT NOT NULL DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS conversations (
    id              SERIAL PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id),
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);

CREATE TABLE IF NOT EXISTS triage_records (
    id              SERIAL PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id),
    patient_id      TEXT,
    triage_score    INTEGER NOT NULL,
    department      TEXT NOT NULL,
    justification   TEXT,
    symptoms        JSONB DEFAULT '[]',
    patient_context JSONB DEFAULT '{}',
    clinician_summary JSONB DEFAULT '{}',
    emergency_detected BOOLEAN DEFAULT FALSE,
    emergency_type  TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_triage_session ON triage_records(session_id);

CREATE TABLE IF NOT EXISTS emergency_incidents (
    id              SERIAL PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id),
    emergency_type  TEXT NOT NULL,
    matched_keyword TEXT,
    raw_message     TEXT,
    alert_sent      BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


def get_schema_sql() -> str:
    """Return the SQL schema for creating tables."""
    return SCHEMA_SQL
