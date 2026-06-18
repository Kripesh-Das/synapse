"""Knowledge retrieval tools — available at summary_generator node."""

from __future__ import annotations

from typing import Any


def search_hospital_kb(query: str, category: str = "", max_results: int = 3) -> dict[str, Any]:
    """Search the hospital knowledge base for protocols and department info."""
    # Placeholder — will be connected to vector DB
    return {
        "results": [],
        "total_found": 0,
        "query": query,
    }


def get_department_info(department_code: str, info_type: str = "all") -> dict[str, Any]:
    """Get detailed information about a hospital department."""
    departments = {
        "ED": {"name": "Emergency Department", "location": "Ground Floor, Building A", "hours": "24/7"},
        "CARD": {"name": "Cardiology", "location": "Building C, 2nd Floor", "hours": "8AM-6PM"},
        "NEURO": {"name": "Neurology", "location": "Building C, 3rd Floor", "hours": "8AM-5PM"},
        "ORTHO": {"name": "Orthopedics", "location": "Building B, 2nd Floor", "hours": "8AM-5PM"},
        "GP": {"name": "General Practice", "location": "Building A, 1st Floor", "hours": "7AM-8PM"},
        "ENT": {"name": "ENT", "location": "Building B, 3rd Floor", "hours": "9AM-5PM"},
        "DERM": {"name": "Dermatology", "location": "Building D, 1st Floor", "hours": "9AM-4PM"},
        "PSYCH": {"name": "Psychiatry", "location": "Building E, 2nd Floor", "hours": "24/7"},
        "GI": {"name": "Gastroenterology", "location": "Building C, 4th Floor", "hours": "8AM-5PM"},
        "OPTH": {"name": "Ophthalmology", "location": "Building D, 3rd Floor", "hours": "9AM-5PM"},
        "URO": {"name": "Urology", "location": "Building B, 4th Floor", "hours": "8AM-5PM"},
        "GYN": {"name": "Gynecology", "location": "Building E, 3rd Floor", "hours": "8AM-5PM"},
    }
    info = departments.get(department_code, {"name": "Unknown", "location": "Unknown", "hours": "Unknown"})
    return {"department": department_code, **info}


def check_wait_times(department_code: str) -> dict[str, Any]:
    """Check current wait times for departments."""
    # Placeholder — will be connected to real-time data
    return {
        "department": department_code,
        "current_wait_minutes": 30,
        "queue_length": 5,
        "trend": "stable",
    }


KNOWLEDGE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_hospital_kb",
            "description": "Search the hospital knowledge base for protocols, department info, or general hospital information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "category": {
                        "type": "string",
                        "enum": ["routing_protocol", "treatment_protocol", "department_info", "general_info", "visitor_policy", "insurance"],
                        "description": "Filter by knowledge category",
                    },
                    "max_results": {"type": "integer", "default": 3, "minimum": 1, "maximum": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_department_info",
            "description": "Get detailed information about a hospital department.",
            "parameters": {
                "type": "object",
                "properties": {
                    "department_code": {
                        "type": "string",
                        "enum": ["ED", "ICU", "CARD", "RESP", "ORTHO", "NEURO", "GI", "DERM", "GP", "OPTH", "ENT", "URO", "GYN", "PSYCH"],
                        "description": "Department code",
                    },
                    "info_type": {"type": "string", "enum": ["location", "hours", "services", "contact", "wait_time", "all"], "default": "all"},
                },
                "required": ["department_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_wait_times",
            "description": "Check current wait times for departments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "department_code": {"type": "string", "description": "Department to check"},
                },
                "required": ["department_code"],
            },
        },
    },
]
