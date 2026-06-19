"""Tests for Medical NER pipeline."""

import pytest
from synapse.extractors.medical_ner import (
    MedicalNER,
    NERResult,
    ExtractedEntity,
    SYMPTOM,
    BODY_PART,
    MEDICATION,
    DURATION,
    SEVERITY,
    ONSET,
    extract_medical_entities,
    extract_symptom_names,
    get_ner,
)


@pytest.fixture(scope="module")
def ner():
    """Shared NER instance for all tests (module scope for performance)."""
    return MedicalNER()


class TestNERInitialization:
    """Test NER pipeline initialization."""

    def test_creates_instance(self):
        n = MedicalNER()
        assert n is not None

    def test_has_entity_ruler(self):
        n = MedicalNER()
        assert "entity_ruler" in n.nlp.pipe_names

    def test_has_matcher(self):
        n = MedicalNER()
        assert n.matcher is not None

    def test_singleton_get_ner(self):
        n1 = get_ner()
        n2 = get_ner()
        assert n1 is n2


class TestSymptomExtraction:
    """Test symptom entity extraction."""

    def test_simple_symptom(self, ner):
        result = ner.extract("I have a headache")
        assert len(result.symptoms) >= 1
        symptom_texts = [s.text.lower() for s in result.symptoms]
        assert "headache" in symptom_texts

    def test_chest_pain(self, ner):
        result = ner.extract("I have chest pain")
        assert len(result.symptoms) >= 1
        symptom_texts = [s.text.lower() for s in result.symptoms]
        assert any("chest pain" in t for t in symptom_texts)

    def test_multiple_symptoms(self, ner):
        result = ner.extract("I have headache and nausea")
        assert len(result.symptoms) >= 2
        symptom_texts = [s.text.lower() for s in result.symptoms]
        assert "headache" in symptom_texts
        assert "nausea" in symptom_texts

    def test_shortness_of_breath(self, ner):
        result = ner.extract("I'm having shortness of breath")
        assert len(result.symptoms) >= 1
        symptom_texts = [s.text.lower() for s in result.symptoms]
        assert any("shortness of breath" in t or "breath" in t for t in symptom_texts)

    def test_dizziness(self, ner):
        result = ner.extract("I have dizziness")
        symptom_texts = [s.text.lower() for s in result.symptoms]
        assert "dizziness" in symptom_texts

    def test_symptom_names_function(self, ner):
        names = ner.extract_symptoms("I have a headache")
        assert "headache" in names

    def test_extract_all_function(self, ner):
        result = ner.extract_all("I have chest pain for 3 days")
        assert "symptoms" in result
        assert len(result["symptoms"]) >= 1


class TestBodyPartExtraction:
    """Test body part entity extraction."""

    def test_chest(self, ner):
        result = ner.extract("pain in my chest")
        bp_texts = [b.text.lower() for b in result.body_parts]
        assert "chest" in bp_texts

    def test_left_arm(self, ner):
        result = ner.extract("my left arm hurts")
        bp_texts = [b.text.lower() for b in result.body_parts]
        assert "left arm" in bp_texts or "arm" in bp_texts

    def test_multiple_body_parts(self, ner):
        result = ner.extract("pain in my chest and back")
        bp_texts = [b.text.lower() for b in result.body_parts]
        assert len(bp_texts) >= 2

    def test_abdomen(self, ner):
        result = ner.extract("pain in my abdomen")
        bp_texts = [b.text.lower() for b in result.body_parts]
        assert "abdomen" in bp_texts


class TestMedicationExtraction:
    """Test medication entity extraction."""

    def test_ibuprofen(self, ner):
        result = ner.extract("I took ibuprofen")
        med_texts = [m.text.lower() for m in result.medications]
        assert "ibuprofen" in med_texts

    def test_taking_metformin(self, ner):
        result = ner.extract("I'm taking metformin daily")
        med_texts = [m.text.lower() for m in result.medications]
        assert "metformin" in med_texts

    def test_multiple_medications(self, ner):
        result = ner.extract("I take aspirin and lisinopril")
        med_texts = [m.text.lower() for m in result.medications]
        assert len(med_texts) >= 2


class TestDurationExtraction:
    """Test duration entity extraction."""

    def test_days(self, ner):
        result = ner.extract("for 3 days")
        dur_texts = [d.text.lower() for d in result.duration]
        assert any("3" in d and "day" in d for d in dur_texts)

    def test_weeks(self, ner):
        result = ner.extract("for about 2 weeks")
        dur_texts = [d.text.lower() for d in result.duration]
        assert len(dur_texts) >= 1

    def test_since_yesterday(self, ner):
        result = ner.extract("since yesterday")
        dur_texts = [d.text.lower() for d in result.duration]
        assert len(dur_texts) >= 1

    def test_few_days(self, ner):
        result = ner.extract("for a few days")
        dur_texts = [d.text.lower() for d in result.duration]
        assert len(dur_texts) >= 1


class TestSeverityExtraction:
    """Test severity entity extraction."""

    def test_numeric_severity(self, ner):
        result = ner.extract("pain is 7 out of 10")
        sev_texts = [s.text.lower() for s in result.severity]
        assert len(result.severity) >= 1

    def test_word_severity(self, ner):
        result = ner.extract("severe pain")
        sev_texts = [s.text.lower() for s in result.severity]
        assert "severe" in sev_texts

    def test_mild_severity(self, ner):
        result = ner.extract("mild headache")
        sev_texts = [s.text.lower() for s in result.severity]
        assert "mild" in sev_texts

    def test_worst_pain(self, ner):
        result = ner.extract("worst pain ever")
        sev_texts = [s.text.lower() for s in result.severity]
        assert len(result.severity) >= 1


class TestOnsetExtraction:
    """Test onset entity extraction."""

    def test_sudden(self, ner):
        result = ner.extract("sudden onset of pain")
        onset_texts = [o.text.lower() for o in result.onset]
        assert "sudden" in onset_texts

    def test_gradual(self, ner):
        result = ner.extract("gradual onset")
        onset_texts = [o.text.lower() for o in result.onset]
        assert "gradual" in onset_texts

    def test_constant(self, ner):
        result = ner.extract("constant pain")
        onset_texts = [o.text.lower() for o in result.onset]
        assert "constant" in onset_texts


class TestNegationDetection:
    """Test negation detection for symptoms."""

    def test_no_headache(self, ner):
        result = ner.extract("no headache")
        negated = [s for s in result.symptoms if s.negated]
        # Should detect "headache" as negated
        assert len(negated) >= 1

    def test_no_chest_pain(self, ner):
        result = ner.extract("no chest pain")
        negated = [s for s in result.symptoms if s.negated]
        assert len(negated) >= 1

    def test_denies_fever(self, ner):
        result = ner.extract("patient denies fever")
        negated = [s for s in result.symptoms if s.negated]
        assert len(negated) >= 1

    def test_without_symptoms(self, ner):
        result = ner.extract("pain without nausea")
        # "nausea" should be negated
        negated = [s for s in result.symptoms if s.negated]
        assert any("nausea" in s.text.lower() for s in negated)

    def test_positive_symptom_not_negated(self, ner):
        result = ner.extract("I have chest pain")
        non_negated = [s for s in result.symptoms if not s.negated]
        assert len(non_negated) >= 1


class TestComplexMessages:
    """Test NER on complex patient messages."""

    def test_full_description(self, ner):
        text = "I've had a severe headache for 3 days, located in my temples. Pain is very bad."
        result = ner.extract(text)
        assert len(result.symptoms) >= 1
        assert len(result.body_parts) >= 1
        assert len(result.duration) >= 1

    def test_multiple_concerns(self, ner):
        text = "I have chest pain and shortness of breath for 2 hours"
        result = ner.extract(text)
        assert len(result.symptoms) >= 2

    def test_medication_with_symptom(self, ner):
        text = "I took ibuprofen for my headache but it didn't help"
        result = ner.extract(text)
        assert len(result.symptoms) >= 1
        assert len(result.medications) >= 1

    def test_with_negation(self, ner):
        text = "I have chest pain but no shortness of breath"
        result = ner.extract(text)
        non_negated = [s for s in result.symptoms if not s.negated]
        negated = [s for s in result.symptoms if s.negated]
        assert len(non_negated) >= 1
        assert len(negated) >= 1


class TestNERResult:
    """Test NERResult data structure."""

    def test_to_dict(self, ner):
        result = ner.extract("I have a headache")
        d = result.to_dict()
        assert "entities" in d
        assert "symptoms" in d
        assert "body_parts" in d
        assert "medications" in d

    def test_extracted_entity_to_dict(self):
        entity = ExtractedEntity(
            text="headache",
            label=SYMPTOM,
            start=0,
            end=8,
            negated=False,
            confidence=1.0,
        )
        d = entity.to_dict()
        assert d["text"] == "headache"
        assert d["label"] == SYMPTOM
        assert d["negated"] is False


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_extract_medical_entities(self):
        result = extract_medical_entities("I have a headache")
        assert isinstance(result, NERResult)
        assert len(result.symptoms) >= 1

    def test_extract_symptom_names(self):
        names = extract_symptom_names("I have a headache and nausea")
        assert "headache" in names
        assert "nausea" in names
