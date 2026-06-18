"""Tests for department routing table."""

from synapse.triage.routing import ROUTING_TABLE, route_department


class TestRouteDepartment:
    def test_high_priority_goes_to_ed(self):
        dept, code = route_department(["chest pain", "shortness of breath"], triage_score=1)
        assert code == "ED"

    def test_urgent_goes_to_ed(self):
        dept, code = route_department(["severe bleeding"], triage_score=2)
        assert code == "ED"

    def test_headache_routes_to_neuro(self):
        dept, code = route_department(["headache", "dizziness"], triage_score=3)
        assert code == "NEURO"

    def test_joint_pain_routes_to_ortho(self):
        dept, code = route_department(["knee pain", "limp"], triage_score=4)
        assert code == "ORTHO"

    def test_skin_issue_routes_to_derm(self):
        dept, code = route_department(["rash", "eczema"], triage_score=4)
        assert code == "DERM"

    def test_stomach_issue_routes_to_gi(self):
        dept, code = route_department(["abdominal pain", "bloating"], triage_score=3)
        assert code == "GI"

    def test_no_match_falls_back_to_gp(self):
        dept, code = route_department(["tiredness"], triage_score=5)
        assert code == "GP"

    def test_empty_symptoms_falls_back_to_gp(self):
        dept, code = route_department([], triage_score=5)
        assert code == "GP"

    def test_multiple_matching_symptoms(self):
        # Should pick the department with most keyword overlaps
        dept, code = route_department(
            ["chest pain", "palpitations", "shortness of breath"],
            triage_score=3,
        )
        assert code == "CARD"


class TestRoutingTableCoverage:
    def test_all_departments_have_keywords(self):
        for rule in ROUTING_TABLE:
            assert len(rule.keywords) > 0, f"{rule.department} has no keywords"

    def test_all_departments_have_codes(self):
        for rule in ROUTING_TABLE:
            assert len(rule.department_code) > 0, f"{rule.department} has no code"

    def test_ed_keywords_include_emergency_terms(self):
        """ED routing rule should cover major emergency terms."""
        ed_rule = next(r for r in ROUTING_TABLE if r.department_code == "ED")
        emergency_terms = ["chest pain", "stroke", "seizure", "gunshot"]
        for term in emergency_terms:
            assert term in ed_rule.keywords, f"ED missing: {term}"
