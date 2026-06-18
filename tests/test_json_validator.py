"""Tests for JSON validator — parse, retry, required fields, patient safety."""

import json

from synapse.validation.json_validator import (
    JSONValidationError,
    parse_llm_json,
    validate_patient_safe,
    validate_triage_score_hidden,
)


class TestParseLlmJson:
    def test_valid_json(self):
        result = parse_llm_json('{"message": "Hello", "next_action": "ask_symptoms"}')
        assert result == {"message": "Hello", "next_action": "ask_symptoms"}

    def test_json_with_code_fences(self):
        raw = '```json\n{"message": "Hello"}\n```'
        result = parse_llm_json(raw)
        assert result == {"message": "Hello"}

    def test_json_with_plain_fences(self):
        raw = '```\n{"message": "Hello"}\n```'
        result = parse_llm_json(raw)
        assert result == {"message": "Hello"}

    def test_whitespace_around_json(self):
        result = parse_llm_json('  \n  {"message": "Hello"}  \n  ')
        assert result == {"message": "Hello"}

    def test_invalid_json_raises_error(self):
        try:
            parse_llm_json("not json at all")
            assert False, "Should have raised JSONValidationError"
        except JSONValidationError as e:
            assert len(e.errors) > 0

    def test_non_dict_json_returns_none(self):
        try:
            parse_llm_json('"just a string"')
            assert False, "Should have raised JSONValidationError"
        except JSONValidationError:
            pass

    def test_array_json_returns_none(self):
        try:
            parse_llm_json('[1, 2, 3]')
            assert False, "Should have raised JSONValidationError"
        except JSONValidationError:
            pass


class TestRequiredFields:
    def test_all_fields_present(self):
        result = parse_llm_json(
            '{"message": "Hi", "next_action": "ask"}',
            required_fields=["message", "next_action"],
        )
        assert result["message"] == "Hi"

    def test_missing_field_raises_error(self):
        try:
            parse_llm_json(
                '{"message": "Hi"}',
                required_fields=["message", "next_action"],
            )
            assert False, "Should have raised JSONValidationError"
        except JSONValidationError:
            pass


class TestRetryWithReprompt:
    def test_retry_succeeds(self):
        call_count = 0

        def reprompt_fn(original, error):
            nonlocal call_count
            call_count += 1
            return '{"message": "Corrected", "next_action": "ask"}'

        result = parse_llm_json(
            "bad json",
            required_fields=["message", "next_action"],
            max_retries=1,
            reprompt_fn=reprompt_fn,
        )
        assert result["message"] == "Corrected"
        assert call_count == 1

    def test_retry_fails_raises_error(self):
        def reprompt_fn(original, error):
            return "still bad json"

        try:
            parse_llm_json(
                "bad json",
                max_retries=1,
                reprompt_fn=reprompt_fn,
            )
            assert False, "Should have raised JSONValidationError"
        except JSONValidationError as e:
            assert len(e.errors) == 2  # Initial + retry

    def test_no_retry_without_reprompt_fn(self):
        try:
            parse_llm_json("bad json", max_retries=3)
            assert False, "Should have raised JSONValidationError"
        except JSONValidationError as e:
            assert len(e.errors) == 1  # Only initial attempt


class TestValidatePatientSafe:
    def test_clean_message(self):
        output = {"message": "Please proceed to the cardiology department."}
        violations = validate_patient_safe(output)
        assert violations == []

    def test_diagnosis_in_message(self):
        output = {"message": "You have pneumonia based on your symptoms."}
        violations = validate_patient_safe(output)
        assert len(violations) > 0

    def test_prescription_in_message(self):
        output = {"message": "You should take this prescription."}
        violations = validate_patient_safe(output)
        assert len(violations) > 0

    def test_triage_score_in_message(self):
        output = {"message": "Your triage score is 2/5."}
        violations = validate_patient_safe(output)
        assert len(violations) > 0

    def test_different_message_key(self):
        output = {"response": "You have a disease."}
        violations = validate_patient_safe(output, patient_message_key="response")
        assert len(violations) > 0

    def test_empty_message(self):
        output = {"message": ""}
        violations = validate_patient_safe(output)
        assert violations == []

    def test_missing_message_key(self):
        output = {}
        violations = validate_patient_safe(output)
        assert violations == []


class TestValidateTriageScoreHidden:
    def test_score_not_leaked(self):
        output = {"message": "Please go to the cardiology department."}
        assert validate_triage_score_hidden(output) is True

    def test_triage_score_leaked(self):
        output = {"message": "Your triage score is 2."}
        assert validate_triage_score_hidden(output) is False

    def test_score_is_pattern(self):
        output = {"message": "The score is 3"}
        assert validate_triage_score_hidden(output) is False

    def test_priority_level_leaked(self):
        output = {"message": "Your priority level is: 2"}
        assert validate_triage_score_hidden(output) is False

    def test_empty_message(self):
        output = {"message": ""}
        assert validate_triage_score_hidden(output) is True

    def test_missing_message(self):
        output = {}
        assert validate_triage_score_hidden(output) is True
