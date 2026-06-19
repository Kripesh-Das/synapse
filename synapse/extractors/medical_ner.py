"""Medical NER pipeline using spaCy EntityRuler and Matcher.

Extracts symptoms, body parts, medications, duration, severity, and onset
from patient messages. Deterministic — no LLM, no hallucination risk.

Architecture:
1. spaCy EntityRuler for phrase matching (symptoms, body parts, medications)
2. spaCy Matcher for contextual patterns (duration, severity, onset)
3. Custom negation detection (replaces medspaCy ConText)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import spacy
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span

# Type aliases
NLP = spacy.language.Language

# Entity labels
SYMPTOM = "SYMPTOM"
BODY_PART = "BODY_PART"
MEDICATION = "MEDICATION"
DURATION = "DURATION"
SEVERITY = "SEVERITY"
ONSET = "ONSET"

# Paths to dictionary files
DICTIONARIES_DIR = Path(__file__).parent / "dictionaries"


@dataclass
class ExtractedEntity:
    """A single extracted medical entity."""
    text: str
    label: str
    start: int
    end: int
    negated: bool = False
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "negated": self.negated,
            "confidence": self.confidence,
        }


@dataclass
class NERResult:
    """Complete NER extraction result."""
    entities: list[ExtractedEntity] = field(default_factory=list)
    symptoms: list[ExtractedEntity] = field(default_factory=list)
    body_parts: list[ExtractedEntity] = field(default_factory=list)
    medications: list[ExtractedEntity] = field(default_factory=list)
    duration: list[ExtractedEntity] = field(default_factory=list)
    severity: list[ExtractedEntity] = field(default_factory=list)
    onset: list[ExtractedEntity] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "symptoms": [e.to_dict() for e in self.symptoms],
            "body_parts": [e.to_dict() for e in self.body_parts],
            "medications": [e.to_dict() for e in self.medications],
            "duration": [e.to_dict() for e in self.duration],
            "severity": [e.to_dict() for e in self.severity],
            "onset": [e.to_dict() for e in self.onset],
        }


def _load_dictionary(filename: str) -> dict[str, list[str]]:
    """Load a JSON dictionary file."""
    filepath = DICTIONARIES_DIR / filename
    if not filepath.exists():
        return {}
    with open(filepath) as f:
        return json.load(f)


def _flatten_dictionary(data: dict[str, list[str]]) -> list[str]:
    """Flatten a nested dictionary into a single list of phrases."""
    phrases = []
    for category_phrases in data.values():
        phrases.extend(category_phrases)
    return phrases


class MedicalNER:
    """Medical Named Entity Recognition pipeline.

    Uses spaCy EntityRuler for phrase matching and custom patterns for
    duration, severity, and onset detection. Includes negation detection.
    """

    def __init__(self, nlp: Optional[NLP] = None):
        """Initialize the NER pipeline.

        Args:
            nlp: Optional pre-loaded spaCy model. If None, loads en_core_web_sm.
        """
        if nlp is None:
            self.nlp = spacy.load("en_core_web_sm")
        else:
            self.nlp = nlp

        # Remove existing entity ruler if present
        if "entity_ruler" in self.nlp.pipe_names:
            self.nlp.remove_pipe("entity_ruler")

        # Add EntityRuler for phrase matching
        self.entity_ruler = self.nlp.add_pipe(
            "entity_ruler", before="ner", config={"overwrite_ents": True}
        )

        # Load dictionaries and add patterns
        self._load_symptom_patterns()
        self._load_body_part_patterns()
        self._load_medication_patterns()

        # Add regex-based patterns for duration, severity, onset
        self._add_regex_patterns()

        # Initialize Matcher for contextual patterns
        self.matcher = Matcher(self.nlp.vocab)
        self._add_contextual_patterns()

        # Negation triggers
        self._negation_words = {
            "no", "not", "none", "neither", "never", "nobody", "nothing",
            "nowhere", "nor", "cannot", "can't", "cant", "don't", "dont",
            "doesn't", "doesnt", "didn't", "didnt", "won't", "wont",
            "wouldn't", "wouldnt", "isn't", "isnt", "aren't", "arent",
            "wasn't", "wasnt", "weren't", "werent", "without", "lack",
            "denies", "denied", "deny", "rules out", "ruled out",
            "no evidence of", "no signs of", "no symptoms of",
            "free of", "absence of",
        }

    def _load_symptom_patterns(self) -> None:
        """Load symptom phrases into EntityRuler."""
        data = _load_dictionary("symptoms.json")
        phrases = _flatten_dictionary(data)
        patterns = [{"label": SYMPTOM, "pattern": phrase} for phrase in phrases]
        self.entity_ruler.add_patterns(patterns)

    def _load_body_part_patterns(self) -> None:
        """Load body part phrases into EntityRuler."""
        data = _load_dictionary("body_parts.json")
        phrases = _flatten_dictionary(data)
        patterns = [{"label": BODY_PART, "pattern": phrase} for phrase in phrases]
        self.entity_ruler.add_patterns(patterns)

    def _load_medication_patterns(self) -> None:
        """Load medication phrases into EntityRuler."""
        data = _load_dictionary("medications.json")
        phrases = _flatten_dictionary(data)
        patterns = [{"label": MEDICATION, "pattern": phrase} for phrase in phrases]
        self.entity_ruler.add_patterns(patterns)

    def _add_regex_patterns(self) -> None:
        """Add regex-based patterns for duration, severity, and onset."""
        # Duration patterns
        duration_patterns = [
            {"label": DURATION, "pattern": [
                {"TEXT": {"REGEX": r"^\d+$"}},
                {"LOWER": {"IN": ["day", "days", "week", "weeks", "month", "months",
                                   "year", "years", "hour", "hours", "minute", "minutes"]}},
            ]},
            {"label": DURATION, "pattern": [
                {"LOWER": "for"},
                {"TEXT": {"REGEX": r"^\d+$"}},
                {"LOWER": {"IN": ["day", "days", "week", "weeks", "month", "months",
                                   "year", "years", "hour", "hours", "minute", "minutes"]}},
            ]},
            {"label": DURATION, "pattern": [
                {"LOWER": {"IN": ["since", "from", "starting"]}},
                {"TEXT": {"REGEX": r"^(yesterday|today|monday|tuesday|wednesday|thursday|friday|saturday|sunday)$"}},
            ]},
            {"label": DURATION, "pattern": [
                {"LOWER": {"IN": ["since", "from", "starting"]}},
                {"TEXT": {"REGEX": r"^\d+$"}},
                {"LOWER": {"IN": ["day", "days", "week", "weeks", "month", "months"]}},
                {"LOWER": "ago"},
            ]},
            {"label": DURATION, "pattern": [
                {"LOWER": "for"},
                {"LOWER": {"IN": ["a", "an"]}},
                {"LOWER": {"IN": ["day", "days", "week", "weeks", "month", "months",
                                   "year", "years", "hour", "hours", "minute", "minutes"]}},
            ]},
            {"label": DURATION, "pattern": [
                {"LOWER": {"IN": ["couple", "few"]}},
                {"LOWER": {"IN": ["day", "days", "week", "weeks", "month", "months"]}},
            ]},
            {"label": DURATION, "pattern": [
                {"LOWER": "about"},
                {"TEXT": {"REGEX": r"^\d+$"}},
                {"LOWER": {"IN": ["day", "days", "week", "weeks", "month", "months"]}},
            ]},
        ]
        self.entity_ruler.add_patterns(duration_patterns)

        # Severity patterns
        severity_patterns = [
            {"label": SEVERITY, "pattern": [
                {"TEXT": {"REGEX": r"^\d+$"}},
                {"TEXT": "/"},
                {"TEXT": {"REGEX": r"^\d+$"}},
            ]},
            {"label": SEVERITY, "pattern": [
                {"TEXT": {"REGEX": r"^\d+$"}},
                {"LOWER": "out"},
                {"LOWER": "of"},
                {"TEXT": {"REGEX": r"^\d+$"}},
            ]},
            {"label": SEVERITY, "pattern": [
                {"LOWER": {"IN": ["mild", "moderate", "severe", "extreme", "terrible",
                                   "horrible", "intense", "sharp", "dull", "throbbing",
                                   "stabbing", "burning", "crushing", "squeezing"]}},
            ]},
            {"label": SEVERITY, "pattern": [
                {"LOWER": {"IN": ["worst", "best"]}},
                {"LOWER": {"IN": ["pain", "headache", "feeling", "symptom"]}},
            ]},
            {"label": SEVERITY, "pattern": [
                {"LOWER": {"IN": ["very", "really", "extremely", "incredibly", "so"]}},
                {"LOWER": {"IN": ["mild", "moderate", "severe", "bad", "painful"]}},
            ]},
            {"label": SEVERITY, "pattern": [
                {"LOWER": {"IN": ["can't", "cant", "cannot"]}},
                {"LOWER": {"IN": ["stand", "tolerate", "bear"]}},
                {"LOWER": {"IN": ["it", "the pain"]}},
            ]},
        ]
        self.entity_ruler.add_patterns(severity_patterns)

        # Onset patterns
        onset_patterns = [
            {"label": ONSET, "pattern": [
                {"LOWER": {"IN": ["sudden", "suddenly", "gradual", "gradually",
                                   "acute", "chronic", "intermittent", "constant",
                                   "persistent", "recurring", "episodic"]}},
            ]},
            {"label": ONSET, "pattern": [
                {"LOWER": "started"},
                {"LOWER": {"IN": ["suddenly", "gradually", "acutely"]}},
            ]},
            {"label": ONSET, "pattern": [
                {"LOWER": {"IN": ["came", "comes"]}},
                {"LOWER": "on"},
                {"LOWER": {"IN": ["suddenly", "gradually"]}},
            ]},
            {"label": ONSET, "pattern": [
                {"LOWER": "get"},
                {"LOWER": {"IN": ["worse", "better"]}},
                {"LOWER": "with"},
            ]},
            {"label": ONSET, "pattern": [
                {"LOWER": "gets"},
                {"LOWER": {"IN": ["worse", "better"]}},
            ]},
        ]
        self.entity_ruler.add_patterns(onset_patterns)

    def _add_contextual_patterns(self) -> None:
        """Add contextual patterns using spaCy Matcher."""
        # Pain in body part
        self.matcher.add("PAIN_IN_BODY", [
            [{"LOWER": {"IN": ["pain", "ache", "hurts", "sore", "tender"]}},
             {"LOWER": "in"},
             {"ENT_TYPE": BODY_PART}],
            [{"LOWER": {"IN": ["pain", "ache", "hurts", "sore", "tender"]}},
             {"LOWER": {"IN": ["on", "at"]}},
             {"ENT_TYPE": BODY_PART}],
        ])

        # Symptom for duration
        self.matcher.add("SYMPTOM_DURATION", [
            [{"ENT_TYPE": SYMPTOM},
             {"LOWER": "for"},
             {"ENT_TYPE": DURATION}],
            [{"ENT_TYPE": SYMPTOM},
             {"ENT_TYPE": DURATION}],
        ])

        # Took medication
        self.matcher.add("TOOK_MEDICATION", [
            [{"LOWER": {"IN": ["took", "taking", "taking", "used", "using", "on"]}},
             {"ENT_TYPE": MEDICATION}],
            [{"ENT_TYPE": MEDICATION},
             {"LOWER": {"IN": ["daily", "twice", "once", "every"]}}],
        ])

        # Severity of symptom
        self.matcher.add("SEVERITY_SYMPTOM", [
            [{"ENT_TYPE": SEVERITY},
             {"LOWER": {"IN": ["pain", "ache", "headache", "discomfort"]}}],
            [{"ENT_TYPE": SEVERITY},
             {"ENT_TYPE": SYMPTOM}],
        ])

    def _detect_negation(self, doc: Doc, entity: Span) -> bool:
        """Detect if an entity is negated.

        Checks for negation words within a 5-token window before the entity.
        """
        entity_start = entity.start

        # Check tokens before the entity (window of 5)
        start_check = max(0, entity_start - 5)
        for i in range(start_check, entity_start):
            token = doc[i]
            # Check single-word negations
            if token.text.lower() in self._negation_words:
                return True
            # Check multi-word negations ("no evidence of", "rules out")
            for neg_phrase in ["no evidence of", "no signs of", "no symptoms of",
                               "rules out", "ruled out", "free of", "absence of"]:
                phrase_tokens = neg_phrase.split()
                phrase_len = len(phrase_tokens)
                if i + phrase_len <= len(doc):
                    doc_slice = doc[i:i + phrase_len]
                    if " ".join(t.text.lower() for t in doc_slice) == neg_phrase:
                        return True

        return False

    def extract(self, text: str) -> NERResult:
        """Extract medical entities from text.

        Args:
            text: Patient message text.

        Returns:
            NERResult with all extracted entities.
        """
        doc = self.nlp(text)
        result = NERResult()

        # Process EntityRuler results
        for ent in doc.ents:
            if ent.label_ in (SYMPTOM, BODY_PART, MEDICATION, DURATION, SEVERITY, ONSET):
                negated = self._detect_negation(doc, ent)
                entity = ExtractedEntity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    negated=negated,
                    confidence=1.0,
                )
                result.entities.append(entity)

                # Categorize
                if ent.label_ == SYMPTOM:
                    result.symptoms.append(entity)
                elif ent.label_ == BODY_PART:
                    result.body_parts.append(entity)
                elif ent.label_ == MEDICATION:
                    result.medications.append(entity)
                elif ent.label_ == DURATION:
                    result.duration.append(entity)
                elif ent.label_ == SEVERITY:
                    result.severity.append(entity)
                elif ent.label_ == ONSET:
                    result.onset.append(entity)

        # Process Matcher results (contextual patterns)
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            match_label = self.nlp.vocab.strings[match_id]
            span = doc[start:end]

            # Add contextual matches if they overlap with existing entities
            # or if they provide new information
            if match_label == "PAIN_IN_BODY":
                # Find body part in the match
                for token in span:
                    if token.ent_type_ == BODY_PART:
                        # Check if body part already extracted
                        bp_text = token.text
                        if not any(e.text == bp_text for e in result.body_parts):
                            negated = self._detect_negation(doc, span)
                            entity = ExtractedEntity(
                                text=bp_text,
                                label=BODY_PART,
                                start=token.idx,
                                end=token.idx + len(token.text),
                                negated=negated,
                                confidence=0.9,
                            )
                            result.entities.append(entity)
                            result.body_parts.append(entity)

            elif match_label == "SYMPTOM_DURATION":
                # Link symptom and duration
                pass  # Already captured by EntityRuler

            elif match_label == "TOOK_MEDICATION":
                # Medication already captured
                pass

            elif match_label == "SEVERITY_SYMPTOM":
                # Severity already captured
                pass

        return result

    def extract_symptoms(self, text: str) -> list[str]:
        """Extract symptom names from text (simplified API).

        Returns:
            List of unique symptom names (lowercased, non-negated).
        """
        result = self.extract(text)
        return list({
            e.text.lower()
            for e in result.symptoms
            if not e.negated
        })

    def extract_all(self, text: str) -> dict:
        """Extract all entities and return as dict (for SessionState).

        Returns:
            Dict with symptoms, body_parts, medications, duration, severity, onset.
        """
        result = self.extract(text)
        return {
            "symptoms": [e.text.lower() for e in result.symptoms if not e.negated],
            "body_parts": [e.text.lower() for e in result.body_parts if not e.negated],
            "medications": [e.text.lower() for e in result.medications],
            "duration": [e.text.lower() for e in result.duration],
            "severity": [e.text.lower() for e in result.severity],
            "onset": [e.text.lower() for e in result.onset],
        }


# Global instance for convenience
_ner_instance: Optional[MedicalNER] = None


def get_ner() -> MedicalNER:
    """Get or create the global NER instance."""
    global _ner_instance
    if _ner_instance is None:
        _ner_instance = MedicalNER()
    return _ner_instance


def extract_medical_entities(text: str) -> NERResult:
    """Convenience function to extract medical entities from text."""
    return get_ner().extract(text)


def extract_symptom_names(text: str) -> list[str]:
    """Convenience function to extract symptom names from text."""
    return get_ner().extract_symptoms(text)
