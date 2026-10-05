import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "ContextBuddy API" in response.json().get("app", "")


def test_analyze_empty_input():
    response = client.post("/api/analyze", json={"source_type": "text", "content": ""})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_analyze_whitespace_input():
    response = client.post("/api/analyze", json={"source_type": "text", "content": "    \n   "})
    assert response.status_code == 400


def test_analyze_valid_conversation():
    sample = """
USER:
I am choosing an MBA program between Wharton and INSEAD.
My budget is $120,000 maximum.
I prefer a program with strong global mobility in Europe.

ASSISTANT:
INSEAD has a 10-month accelerated format and exceptional European alumni reach.
Wharton offers deep US finance prestige.

USER:
I've decided to choose INSEAD because of the 1-year timeline and Paris campus.
I've ruled out Wharton due to the higher 2-year tuition and living costs.
Should I apply in Round 1 or Round 2?
"""
    response = client.post("/api/analyze", json={"source_type": "text", "content": sample})
    assert response.status_code == 200
    data = response.json()

    # Verify structured fields
    assert "primary_goal" in data
    assert "decisions" in data
    assert "rejected_ideas" in data
    assert "constraints" in data
    assert "open_questions" in data
    assert "portable_context" in data
    assert "metrics" in data

    # Verify content accuracy
    assert any("insead" in d.lower() for d in data["decisions"])
    assert any("wharton" in r.lower() for r in data["rejected_ideas"])
    assert data["metrics"]["original_words"] > 0
    assert data["metrics"]["compressed_words"] > 0
    assert data["metrics"]["compression_percentage"] >= 0
