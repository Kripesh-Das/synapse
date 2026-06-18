"""Comprehensive tests for the triage rules engine.

Tests all scoring rules, age adjustments, pregnancy adjustments, edge cases.
"""

from synapse.state import Symptom
from synapse.triage.rules_engine import (
    CRITICAL_SYMPTOMS,
    HIGH_SEVERITY_SYMPTOMS,
    TriageRulesEngine,
)


class TestCriticalSymptoms:
    """Symptoms in CRITICAL_SYMPTOMS should always score 1."""

    def setup_method(self):
        self.engine = TriageRulesEngine()

    def test_chest_pain_scores_1(self):
        result = self.engine.compute_triage([Symptom(name="chest pain")])
        assert result.score == 1
        assert any("Critical symptom" in j for j in result.justifications)

    def test_stroke_scores_1(self):
        result = self.engine.compute_triage([Symptom(name="stroke")])
        assert result.score == 1

    def test_seizure_scores_1(self):
        result = self.engine.compute_triage([Symptom(name="seizure")])
        assert result.score == 1

    def test_unconscious_scores_1(self):
        result = self.engine.compute_triage([Symptom(name="unconscious")])
        assert result.score == 1

    def test_severe_bleeding_scores_1(self):
        result = self.engine.compute_triage([Symptom(name="severe bleeding")])
        assert result.score == 1

    def test_anaphylaxis_scores_1(self):
        result = self.engine.compute_triage([Symptom(name="anaphylaxis")])
        assert result.score == 1

    def test_suicide_scores_1(self):
        result = self.engine.compute_triage([Symptom(name="suicide")])
        assert result.score == 1

    def test_overdose_scores_1(self):
        result = self.engine.compute_triage([Symptom(name="overdose")])
        assert result.score == 1


class TestHighSeveritySymptoms:
    """Symptoms in HIGH_SEVERITY_SYMPTOMS should score at most 2."""

    def setup_method(self):
        self.engine = TriageRulesEngine()

    def test_severe_headache_scores_2(self):
        result = self.engine.compute_triage([Symptom(name="severe headache")])
        assert result.score == 2

    def test_high_fever_scores_2(self):
        result = self.engine.compute_triage([Symptom(name="high fever")])
        assert result.score == 2

    def test_vomiting_blood_scores_2(self):
        result = self.engine.compute_triage([Symptom(name="vomiting blood")])
        assert result.score == 2


class TestSeverityRating:
    """Severity rating (1-10) affects scoring."""

    def setup_method(self):
        self.engine = TriageRulesEngine()

    def test_severity_10(self):
        result = self.engine.compute_triage([Symptom(name="headache", severity=10)])
        assert result.score == 2
        assert any("10/10" in j for j in result.justifications)

    def test_severity_8(self):
        result = self.engine.compute_triage([Symptom(name="headache", severity=8)])
        assert result.score == 2

    def test_severity_7(self):
        result = self.engine.compute_triage([Symptom(name="headache", severity=7)])
        assert result.score == 3

    def test_severity_6(self):
        result = self.engine.compute_triage([Symptom(name="headache", severity=6)])
        assert result.score == 3

    def test_severity_5(self):
        result = self.engine.compute_triage([Symptom(name="headache", severity=5)])
        assert result.score == 5  # No severity rule fires, stays at default

    def test_severity_1(self):
        result = self.engine.compute_triage([Symptom(name="headache", severity=1)])
        assert result.score == 5


class TestSuddenOnset:
    """Sudden onset should upgrade urgency."""

    def setup_method(self):
        self.engine = TriageRulesEngine()

    def test_sudden_onset_field(self):
        result = self.engine.compute_triage(
            [Symptom(name="headache", onset="sudden")]
        )
        assert result.score == 2
        assert any("Sudden onset" in j for j in result.justifications)

    def test_sudden_in_duration(self):
        result = self.engine.compute_triage(
            [Symptom(name="headache", duration="just happened")]
        )
        assert result.score == 2

    def test_immediate_onset(self):
        result = self.engine.compute_triage(
            [Symptom(name="dizziness", onset="immediate")]
        )
        assert result.score == 2


class TestAgeAdjustments:
    def setup_method(self):
        self.engine = TriageRulesEngine()

    def test_elderly_upgrades_urgency(self):
        # Headache alone = 5, elderly = min(5, max(1, 5-1)) = 4
        result = self.engine.compute_triage(
            [Symptom(name="headache")], age=70
        )
        assert result.score == 4
        assert any("Elderly" in j for j in result.justifications)

    def test_child_upgrades_urgency(self):
        result = self.engine.compute_triage(
            [Symptom(name="fever")], age=3
        )
        assert result.score == 4
        assert any("child" in j.lower() for j in result.justifications)

    def test_elderly_with_critical_stays_1(self):
        result = self.engine.compute_triage(
            [Symptom(name="chest pain")], age=80
        )
        assert result.score == 1  # Can't go below 1

    def test_adult_no_adjustment(self):
        result = self.engine.compute_triage(
            [Symptom(name="headache")], age=30
        )
        assert result.score == 5


class TestPregnancyAdjustment:
    def setup_method(self):
        self.engine = TriageRulesEngine()

    def test_pregnant_upgrades_urgency(self):
        result = self.engine.compute_triage(
            [Symptom(name="abdominal pain")], pregnancy_status=True
        )
        assert result.score == 4
        assert any("Pregnant" in j for j in result.justifications)

    def test_pregnant_with_critical_stays_1(self):
        result = self.engine.compute_triage(
            [Symptom(name="chest pain")], pregnancy_status=True
        )
        assert result.score == 1


class TestMultipleSymptoms:
    def setup_method(self):
        self.engine = TriageRulesEngine()

    def test_three_symptoms_upgrade(self):
        result = self.engine.compute_triage([
            Symptom(name="headache"),
            Symptom(name="nausea"),
            Symptom(name="fatigue"),
        ])
        assert result.score == 3
        assert any("Multiple symptoms" in j for j in result.justifications)

    def test_two_symptoms_no_multi_upgrade(self):
        result = self.engine.compute_triage([
            Symptom(name="headache"),
            Symptom(name="nausea"),
        ])
        # Only 2 symptoms, no multi-symptom rule
        assert result.score == 5


class TestCombinedRules:
    """Multiple rules firing together."""

    def setup_method(self):
        self.engine = TriageRulesEngine()

    def test_critical_plus_high_severity(self):
        # Critical symptom scores 1, severity 9 also scores 2
        result = self.engine.compute_triage([
            Symptom(name="chest pain", severity=9)
        ])
        assert result.score == 1  # Critical wins

    def test_moderate_symptom_with_high_severity(self):
        result = self.engine.compute_triage([
            Symptom(name="headache", severity=8)
        ])
        assert result.score == 2  # Severity 8 triggers

    def test_elderly_with_sudden_onset(self):
        result = self.engine.compute_triage(
            [Symptom(name="dizziness", onset="sudden")], age=72
        )
        assert result.score == 1  # Sudden onset = 2, elderly upgrades to 1


class TestDepartmentRouting:
    def setup_method(self):
        self.engine = TriageRulesEngine()

    def test_critical_goes_to_ed(self):
        result = self.engine.compute_triage([Symptom(name="chest pain")])
        assert result.department == "ED"

    def test_headache_goes_to_neuro(self):
        result = self.engine.compute_triage([Symptom(name="headache")])
        assert result.department == "NEURO"

    def test_joint_pain_goes_to_ortho(self):
        result = self.engine.compute_triage([Symptom(name="knee pain")])
        assert result.department == "ORTHO"

    def test_general_symptom_goes_to_gp(self):
        result = self.engine.compute_triage([Symptom(name="tiredness")])
        assert result.department == "GP"


class TestWaitTimes:
    def setup_method(self):
        self.engine = TriageRulesEngine()

    def test_ed_critical_wait_is_zero(self):
        result = self.engine.compute_triage([Symptom(name="chest pain")])
        assert result.estimated_wait == 0

    def test_gp_moderate_wait(self):
        result = self.engine.compute_triage([Symptom(name="tiredness")])
        assert result.estimated_wait == 180


class TestNoSymptoms:
    def setup_method(self):
        self.engine = TriageRulesEngine()

    def test_empty_symptoms(self):
        result = self.engine.compute_triage([])
        assert result.score == 5
        assert result.department == "General Practice"
        assert any("No symptoms" in j for j in result.justifications)


class TestCustomWaitTimes:
    def test_custom_wait_times(self):
        custom = {"GP": {5: 30}}
        engine = TriageRulesEngine(wait_times=custom)
        result = engine.compute_triage([Symptom(name="tiredness")])
        assert result.estimated_wait == 30
