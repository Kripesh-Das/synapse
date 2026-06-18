"""Comprehensive tests for emergency keyword detector.

Covers all emergency categories, edge cases, and negative cases.
This is the most safety-critical module — tests must be thorough.
"""

from synapse.extractors.emergency_detector import (
    EMERGENCY_KEYWORDS,
    EmergencyResult,
    detect_emergency,
)


class TestCardiacEmergency:
    def test_chest_pain(self):
        result = detect_emergency("I have chest pain")
        assert result.detected is True
        assert result.emergency_type == "cardiac"
        assert result.confidence == "rule_based_100"

    def test_heart_attack(self):
        result = detect_emergency("I think I'm having a heart attack")
        assert result.detected is True
        assert result.emergency_type == "cardiac"

    def test_cant_breathe(self):
        result = detect_emergency("I can't breathe")
        assert result.detected is True
        assert result.emergency_type == "cardiac"

    def test_cant_breathe_apostrophe_variant(self):
        result = detect_emergency("I cant breathe")
        assert result.detected is True
        assert result.emergency_type == "cardiac"

    def test_shortness_of_breath(self):
        result = detect_emergency("Having shortness of breath since morning")
        assert result.detected is True
        assert result.emergency_type == "cardiac"

    def test_chest_tightness(self):
        result = detect_emergency("My chest feels tight")
        assert result.detected is True
        assert result.emergency_type == "cardiac"

    def test_crushing_chest_pain(self):
        result = detect_emergency("There is a crushing chest pain")
        assert result.detected is True
        assert result.emergency_type == "cardiac"

    def test_pain_radiating_to_arm(self):
        result = detect_emergency("chest pain radiating to arm")
        assert result.detected is True
        assert result.emergency_type == "cardiac"

    def test_pain_radiating_to_jaw(self):
        result = detect_emergency("pain radiating to jaw")
        assert result.detected is True
        assert result.emergency_type == "cardiac"


class TestNeurologicalEmergency:
    def test_unconscious(self):
        result = detect_emergency("My father is unconscious")
        assert result.detected is True
        assert result.emergency_type == "neurological"

    def test_seizure(self):
        result = detect_emergency("She had a seizure")
        assert result.detected is True
        assert result.emergency_type == "neurological"

    def test_stroke(self):
        result = detect_emergency("I think it's a stroke")
        assert result.detected is True
        assert result.emergency_type == "neurological"

    def test_cant_move(self):
        result = detect_emergency("I can't move my left side")
        assert result.detected is True
        assert result.emergency_type == "neurological"

    def test_paralyzed(self):
        result = detect_emergency("My arm is paralyzed")
        assert result.detected is True
        assert result.emergency_type == "neurological"

    def test_severe_headache(self):
        result = detect_emergency("Worst headache of my life")
        assert result.detected is True
        assert result.emergency_type == "neurological"

    def test_slurred_speech(self):
        result = detect_emergency("My speech is slurred")
        assert result.detected is True
        assert result.emergency_type == "neurological"

    def test_face_drooping(self):
        result = detect_emergency("My face is drooping on one side")
        assert result.detected is True
        assert result.emergency_type == "neurological"


class TestTraumaEmergency:
    def test_severe_bleeding(self):
        result = detect_emergency("There is severe bleeding")
        assert result.detected is True
        assert result.emergency_type == "trauma"

    def test_bleeding_heavily(self):
        result = detect_emergency("I'm bleeding heavily from the leg")
        assert result.detected is True
        assert result.emergency_type == "trauma"

    def test_gunshot(self):
        result = detect_emergency("Gunshot wound to the abdomen")
        assert result.detected is True
        assert result.emergency_type == "trauma"

    def test_stabbed(self):
        result = detect_emergency("He was stabbed")
        assert result.detected is True
        assert result.emergency_type == "trauma"

    def test_broken_bone_protruding(self):
        result = detect_emergency("broken bone protruding from arm")
        assert result.detected is True
        assert result.emergency_type == "trauma"

    def test_car_crash(self):
        result = detect_emergency("Was in a major car crash")
        assert result.detected is True
        assert result.emergency_type == "trauma"

    def test_head_injury(self):
        result = detect_emergency("Hit my head really hard")
        assert result.detected is True
        assert result.emergency_type == "trauma"

    def test_fall_from_height(self):
        result = detect_emergency("Fell from a height of 10 feet")
        assert result.detected is True
        assert result.emergency_type == "trauma"


class TestPsychiatricEmergency:
    def test_suicide(self):
        result = detect_emergency("I want to commit suicide")
        assert result.detected is True
        assert result.emergency_type == "psychiatric"

    def test_kill_myself(self):
        result = detect_emergency("I'm going to kill myself")
        assert result.detected is True
        assert result.emergency_type == "psychiatric"

    def test_want_to_die(self):
        result = detect_emergency("I just want to die")
        assert result.detected is True
        assert result.emergency_type == "psychiatric"

    def test_overdose(self):
        result = detect_emergency("I took an overdose of pills")
        assert result.detected is True
        assert result.emergency_type == "psychiatric"

    def test_self_harm(self):
        result = detect_emergency("I've been hurting myself")
        assert result.detected is True
        assert result.emergency_type == "psychiatric"


class TestObstetricEmergency:
    def test_pregnant_and_bleeding(self):
        result = detect_emergency("I'm pregnant and bleeding")
        assert result.detected is True
        assert result.emergency_type == "obstetric"

    def test_labor_pains(self):
        result = detect_emergency("Having labor pains at 30 weeks")
        assert result.detected is True
        assert result.emergency_type == "obstetric"

    def test_water_broke(self):
        result = detect_emergency("My water broke early")
        assert result.detected is True
        assert result.emergency_type == "obstetric"


class TestRespiratoryEmergency:
    def test_choking(self):
        result = detect_emergency("I'm choking")
        assert result.detected is True
        assert result.emergency_type == "respiratory"

    def test_anaphylaxis(self):
        result = detect_emergency("I think it's anaphylaxis")
        assert result.detected is True
        assert result.emergency_type == "respiratory"

    def test_throat_closing(self):
        result = detect_emergency("My throat is closing up")
        assert result.detected is True
        assert result.emergency_type == "respiratory"

    def test_severe_allergic_reaction(self):
        result = detect_emergency("Having a severe allergic reaction")
        assert result.detected is True
        assert result.emergency_type == "respiratory"

    def test_tongue_swelling(self):
        result = detect_emergency("My tongue is swelling")
        assert result.detected is True
        assert result.emergency_type == "respiratory"


class TestNegativeCases:
    """Messages that should NOT trigger emergency detection."""

    def test_normal_headache(self):
        result = detect_emergency("I have a mild headache")
        assert result.detected is False

    def test_fever(self):
        result = detect_emergency("I've had a fever for 2 days")
        assert result.detected is False

    def test_cough(self):
        result = detect_emergency("I've been coughing a lot")
        assert result.detected is False

    def test_stomach_pain(self):
        result = detect_emergency("My stomach hurts")
        assert result.detected is False

    def test_back_pain(self):
        result = detect_emergency("I have lower back pain")
        assert result.detected is False

    def test_joint_pain(self):
        result = detect_emergency("My knee hurts when I walk")
        assert result.detected is False

    def test_rash(self):
        result = detect_emergency("I have a rash on my arm")
        assert result.detected is False

    def test_fatigue(self):
        result = detect_emergency("I feel tired all the time")
        assert result.detected is False

    def test_empty_message(self):
        result = detect_emergency("")
        assert result.detected is False

    def test_just_greeting(self):
        result = detect_emergency("Hello, I need help")
        assert result.detected is False


class TestCaseInsensitivity:
    def test_uppercase(self):
        result = detect_emergency("CHEST PAIN")
        assert result.detected is True
        assert result.emergency_type == "cardiac"

    def test_mixed_case(self):
        result = detect_emergency("I have Chest Pain and Shortness Of Breath")
        assert result.detected is True
        assert result.emergency_type == "cardiac"

    def test_partial_capitalization(self):
        result = detect_emergency("SEIZURE just happened")
        assert result.detected is True
        assert result.emergency_type == "neurological"


class TestPriorityOrdering:
    """When a message contains multiple emergency types, the highest priority wins."""

    def test_cardiac_over_neurological(self):
        # Cardiac comes first in EMERGENCY_KEYWORDS dict
        result = detect_emergency("chest pain and slurred speech")
        assert result.emergency_type == "cardiac"

    def test_trauma_over_respiratory(self):
        result = detect_emergency("severe bleeding and choking")
        assert result.emergency_type == "trauma"


class TestEdgeCases:
    def test_message_with_ambulance_context(self):
        result = detect_emergency("The ambulance brought him, he had a seizure")
        assert result.detected is True
        assert result.emergency_type == "neurological"

    def test_child_describing_parent(self):
        result = detect_emergency("My mom says she has chest pain")
        assert result.detected is True
        assert result.emergency_type == "cardiac"

    def test_symptom_with_context(self):
        result = detect_emergency(
            "I'm a 45 year old male, I've been having chest pain "
            "for about 2 hours now, it's getting worse"
        )
        assert result.detected is True
        assert result.emergency_type == "cardiac"


class TestKeywordCoverage:
    """Verify all defined keywords are actually testable."""

    def test_all_categories_have_keywords(self):
        for category, keywords in EMERGENCY_KEYWORDS.items():
            assert len(keywords) > 0, f"Category {category} has no keywords"

    def test_all_categories_detected(self):
        """Each emergency category should have at least one keyword that triggers detection."""
        for category, keywords in EMERGENCY_KEYWORDS.items():
            result = detect_emergency(keywords[0])
            assert result.detected is True, (
                f"Category {category} keyword '{keywords[0]}' did not trigger detection"
            )
            assert result.emergency_type == category
