"""Tests for PostgreSQL persistence repository — uses mocked connections."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from synapse.db.schema import get_schema_sql
from synapse.state import Symptom, TriageResult


class TestSchema:
    def test_schema_has_sessions_table(self):
        sql = get_schema_sql()
        assert "CREATE TABLE IF NOT EXISTS sessions" in sql

    def test_schema_has_conversations_table(self):
        sql = get_schema_sql()
        assert "CREATE TABLE IF NOT EXISTS conversations" in sql

    def test_schema_has_triage_records_table(self):
        sql = get_schema_sql()
        assert "CREATE TABLE IF NOT EXISTS triage_records" in sql

    def test_schema_has_emergency_incidents_table(self):
        sql = get_schema_sql()
        assert "CREATE TABLE IF NOT EXISTS emergency_incidents" in sql

    def test_schema_has_indexes(self):
        sql = get_schema_sql()
        assert "idx_conversations_session" in sql
        assert "idx_triage_session" in sql


class TestRepositoryInserts:
    """Test repository functions with mocked DB connections."""

    @patch("synapse.db.repository.get_connection")
    def test_insert_session(self, mock_get_conn):
        from synapse.db.repository import insert_session

        mock_conn = MagicMock()
        mock_get_conn.return_value.__enter__.return_value = mock_conn

        insert_session("sess-123", patient_id="PT-001", source="web", language="en")

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        assert "sess-123" in call_args[0][1]
        mock_conn.commit.assert_called_once()

    @patch("synapse.db.repository.get_connection")
    def test_insert_conversation_message(self, mock_get_conn):
        from synapse.db.repository import insert_conversation_message

        mock_conn = MagicMock()
        mock_get_conn.return_value.__enter__.return_value = mock_conn

        insert_conversation_message("sess-123", "user", "I have chest pain")

        mock_conn.execute.assert_called_once()
        mock_conn.commit.assert_called_once()

    @patch("synapse.db.repository.get_connection")
    def test_insert_triage_record(self, mock_get_conn):
        from synapse.db.repository import insert_triage_record

        mock_conn = MagicMock()
        mock_get_conn.return_value.__enter__.return_value = mock_conn

        triage = TriageResult(score=2, justifications=["Critical symptom"], department="ED")
        symptoms = [Symptom(name="chest pain", severity=9)]

        insert_triage_record(
            session_id="sess-123",
            patient_id="PT-001",
            triage_result=triage,
            symptoms=symptoms,
            patient_context={"age": 45},
            clinician_summary={"chief_complaint": "chest pain"},
            emergency_detected=True,
            emergency_type="cardiac",
        )

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        # Verify triage score is in the parameters
        params = call_args[0][1]
        assert 2 in params  # triage_score
        assert "ED" in params  # department

    @patch("synapse.db.repository.get_connection")
    def test_insert_emergency_incident(self, mock_get_conn):
        from synapse.db.repository import insert_emergency_incident

        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = (1,)
        mock_conn.execute.return_value = mock_result
        mock_get_conn.return_value.__enter__.return_value = mock_conn

        incident_id = insert_emergency_incident(
            session_id="sess-123",
            emergency_type="cardiac",
            matched_keyword="chest pain",
            raw_message="I have chest pain",
        )

        assert incident_id == 1
        mock_conn.commit.assert_called_once()

    @patch("synapse.db.repository.get_connection")
    def test_close_session(self, mock_get_conn):
        from synapse.db.repository import close_session

        mock_conn = MagicMock()
        mock_get_conn.return_value.__enter__.return_value = mock_conn

        close_session("sess-123")

        mock_conn.execute.assert_called_once()
        mock_conn.commit.assert_called_once()


class TestRepositoryQueries:
    @patch("synapse.db.repository.get_connection")
    def test_get_session_found(self, mock_get_conn):
        from synapse.db.repository import get_session

        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = {"session_id": "sess-123", "source": "web"}
        mock_conn.execute.return_value = mock_result
        mock_get_conn.return_value.__enter__.return_value = mock_conn

        result = get_session("sess-123")
        assert result is not None
        assert result["session_id"] == "sess-123"

    @patch("synapse.db.repository.get_connection")
    def test_get_session_not_found(self, mock_get_conn):
        from synapse.db.repository import get_session

        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_conn.execute.return_value = mock_result
        mock_get_conn.return_value.__enter__.return_value = mock_conn

        result = get_session("nonexistent")
        assert result is None
