from __future__ import annotations

from dataclasses import dataclass, field

from synapse.state import Symptom, TriageResult


# Critical symptoms that always score 1 (immediate)
CRITICAL_SYMPTOMS = frozenset({
    "chest pain", "heart attack", "stroke", "seizure",
    "unconscious", "severe bleeding", "anaphylaxis",
    "difficulty breathing", "suicide", "overdose",
})

# High-severity symptoms that score at most 2
HIGH_SEVERITY_SYMPTOMS = frozenset({
    "severe headache", "high fever", "vomiting blood",
    "blood in urine", "blood in stool", "abdominal pain severe",
    "palpitations", "loss of consciousness",
})

# Sudden-onset keywords
SUDDEN_ONSET_KEYWORDS = frozenset({
    "sudden", "immediate", "seconds", "just happened",
    "right now", "acute", "all of a sudden",
})


@dataclass
class TriageRule:
    name: str
    description: str = ""


class TriageRulesEngine:
    """Deterministic triage scoring engine.

    NO LLM involvement. Same symptoms always produce the same score.
    Score: 1 (critical) to 5 (low urgency).
    """

    def __init__(self, wait_times: dict | None = None):
        # department_code -> {score: wait_minutes}
        self.wait_times = wait_times or {
            "ED": {1: 0, 2: 15, 3: 30, 4: 60, 5: 120},
            "CARD": {1: 0, 2: 20, 3: 45, 4: 90, 5: 180},
            "NEURO": {1: 0, 2: 20, 3: 45, 4: 90, 5: 180},
            "ORTHO": {1: 0, 2: 30, 3: 60, 4: 120, 5: 240},
            "GP": {1: 0, 2: 30, 3: 60, 4: 120, 5: 180},
            "ENT": {1: 0, 2: 30, 3: 60, 4: 120, 5: 180},
            "DERM": {1: 0, 2: 45, 3: 90, 4: 180, 5: 360},
            "PSYCH": {1: 0, 2: 20, 3: 45, 4: 90, 5: 180},
            "GI": {1: 0, 2: 30, 3: 60, 4: 120, 5: 240},
            "OPTH": {1: 0, 2: 30, 3: 60, 4: 120, 5: 240},
            "URO": {1: 0, 2: 30, 3: 60, 4: 120, 5: 240},
            "GYN": {1: 0, 2: 30, 3: 60, 4: 120, 5: 240},
        }

    def compute_triage(
        self,
        symptoms: list[Symptom],
        age: int = 0,
        pregnancy_status: bool = False,
    ) -> TriageResult:
        """Compute triage score and justifications from extracted symptoms.

        Returns a TriageResult with score (1-5), justifications list, and department.
        """
        score = 5  # Start at lowest urgency
        justifications: list[str] = []

        if not symptoms:
            justifications.append("No symptoms reported")
            return TriageResult(
                score=score,
                justifications=justifications,
                department="General Practice",
                estimated_wait=self.wait_times.get("GP", {}).get(5, 180),
            )

        for symptom in symptoms:
            name_lower = symptom.name.lower()

            # Rule 1: Critical symptoms
            if name_lower in CRITICAL_SYMPTOMS:
                score = min(score, 1)
                justifications.append(f"Critical symptom: {symptom.name}")

            # Rule 2: High-severity symptoms
            elif name_lower in HIGH_SEVERITY_SYMPTOMS:
                score = min(score, 2)
                justifications.append(f"High-severity symptom: {symptom.name}")

            # Rule 3: Severity rating (1-10 scale)
            if symptom.severity >= 8:
                score = min(score, 2)
                justifications.append(f"High severity rating: {symptom.severity}/10")
            elif symptom.severity >= 6:
                score = min(score, 3)
                justifications.append(f"Moderate-high severity: {symptom.severity}/10")

            # Rule 4: Sudden onset
            onset_lower = symptom.onset.lower()
            duration_lower = symptom.duration.lower()
            if any(kw in onset_lower for kw in SUDDEN_ONSET_KEYWORDS):
                score = min(score, 2)
                justifications.append("Sudden onset")
            elif any(kw in duration_lower for kw in SUDDEN_ONSET_KEYWORDS):
                score = min(score, 2)
                justifications.append("Sudden onset (from duration)")

            # Rule 5: Multiple symptoms (comorbidity)
            if len(symptoms) >= 3:
                score = min(score, 3)
                justifications.append(f"Multiple symptoms ({len(symptoms)} reported)")

        # Age-based adjustments
        if age > 65:
            score = min(score, max(1, score - 1))
            justifications.append("Elderly patient (age > 65) — upgraded urgency")
        elif age < 5 and age > 0:
            score = min(score, max(1, score - 1))
            justifications.append("Young child (age < 5) — upgraded urgency")

        # Pregnancy adjustment
        if pregnancy_status:
            score = min(score, max(1, score - 1))
            justifications.append("Pregnant patient — upgraded urgency")

        # Determine department (simplified — full routing in triage/routing.py)
        department = self._default_department(score, symptoms)

        return TriageResult(
            score=score,
            justifications=justifications,
            department=department,
            estimated_wait=self.wait_times.get(department, {}).get(score, 60),
        )

    def _default_department(self, score: int, symptoms: list[Symptom]) -> str:
        """Simple department assignment for initial implementation."""
        if score <= 2:
            return "ED"

        all_names = " ".join(s.name.lower() for s in symptoms)

        if any(kw in all_names for kw in ["chest", "heart", "palpitation"]):
            return "CARD"
        if any(kw in all_names for kw in ["headache", "numbness", "dizziness", "seizure"]):
            return "NEURO"
        if any(kw in all_names for kw in ["bone", "joint", "fracture", "sprain", "back pain", "knee pain", "shoulder pain"]):
            return "ORTHO"
        if any(kw in all_names for kw in ["ear", "throat", "sinus", "nose"]):
            return "ENT"
        if any(kw in all_names for kw in ["skin", "rash", "eczema"]):
            return "DERM"
        if any(kw in all_names for kw in ["stomach", "abdominal", "nausea", "vomit"]):
            return "GI"

        return "GP"
