from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RoutingRule:
    department: str
    department_code: str
    keywords: list[str]
    description: str = ""


# Department routing table — maps symptom keywords to departments.
# Used by the triage engine after scoring to determine placement.
ROUTING_TABLE: list[RoutingRule] = [
    RoutingRule(
        department="Emergency Department",
        department_code="ED",
        keywords=[
            "chest pain", "heart attack", "stroke", "seizure",
            "unconscious", "severe bleeding", "gunshot", "stabbed",
            "difficulty breathing", "anaphylaxis", "overdose",
            "suicide", "major trauma",
        ],
        description="Life-threatening emergencies",
    ),
    RoutingRule(
        department="Cardiology",
        department_code="CARD",
        keywords=[
            "chest pain", "palpitations", "heart racing",
            "irregular heartbeat", "high blood pressure",
            "shortness of breath", "swollen legs",
        ],
        description="Heart and cardiovascular issues",
    ),
    RoutingRule(
        department="Neurology",
        department_code="NEURO",
        keywords=[
            "headache", "migraine", "dizziness", "numbness",
            "tingling", "memory loss", "confusion", "seizure",
            "balance problems", "tremor",
        ],
        description="Brain and nervous system",
    ),
    RoutingRule(
        department="Orthopedics",
        department_code="ORTHO",
        keywords=[
            "broken bone", "fracture", "sprain", "joint pain",
            "back pain", "neck pain", "knee pain", "shoulder pain",
            "limp", "swollen joint",
        ],
        description="Bones, joints, and muscles",
    ),
    RoutingRule(
        department="General Practice",
        department_code="GP",
        keywords=[
            "fever", "cough", "cold", "flu", "sore throat",
            "stomach ache", "nausea", "vomiting", "diarrhea",
            "rash", "fatigue", "general checkup",
        ],
        description="General and non-urgent issues",
    ),
    RoutingRule(
        department="ENT",
        department_code="ENT",
        keywords=[
            "ear pain", "hearing loss", "sore throat",
            "tonsils", "sinus", "nosebleed", "throat pain",
        ],
        description="Ear, nose, and throat",
    ),
    RoutingRule(
        department="Dermatology",
        department_code="DERM",
        keywords=[
            "rash", "skin rash", "acne", "eczema",
            "skin lesion", "mole change", "hives",
        ],
        description="Skin conditions",
    ),
    RoutingRule(
        department="Psychiatry",
        department_code="PSYCH",
        keywords=[
            "anxiety", "depression", "panic attack",
            "insomnia", "mood swing", "hallucination",
            "hearing voices", "paranoia",
        ],
        description="Mental health",
    ),
    RoutingRule(
        department="Gastroenterology",
        department_code="GI",
        keywords=[
            "stomach pain", "abdominal pain", "bloating",
            "heartburn", "acid reflux", "blood in stool",
            "constipation", "bowel problems",
        ],
        description="Digestive system",
    ),
    RoutingRule(
        department="Ophthalmology",
        department_code="OPTH",
        keywords=[
            "eye pain", "vision loss", "blurred vision",
            "eye redness", "flashing lights", "floaters",
        ],
        description="Eye conditions",
    ),
    RoutingRule(
        department="Urology",
        department_code="URO",
        keywords=[
            "painful urination", "blood in urine",
            "frequent urination", "kidney stone",
            "back pain side",
        ],
        description="Urinary tract and kidneys",
    ),
    RoutingRule(
        department="Gynecology",
        department_code="GYN",
        keywords=[
            "pelvic pain", "irregular period", "heavy bleeding",
            "vaginal discharge", "pain during intercourse",
        ],
        description="Female reproductive health",
    ),
]


def route_department(symptoms: list[str], triage_score: int) -> tuple[str, str]:
    """Determine department based on symptom keywords and triage score.

    Returns (department_name, department_code).
    Falls back to General Practice if no match found.
    """
    if triage_score <= 2:
        return "Emergency Department", "ED"

    symptoms_text = " ".join(symptoms).lower()

    best_match: RoutingRule | None = None
    best_overlap = 0

    for rule in ROUTING_TABLE:
        overlap = sum(1 for kw in rule.keywords if kw in symptoms_text)
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = rule

    if best_match and best_overlap > 0:
        return best_match.department, best_match.department_code

    return "General Practice", "GP"
