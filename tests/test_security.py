
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint_remains_public():
    response = client.get("/api/health")

    assert response.status_code == 200


def test_documents_endpoint_rejects_missing_api_key():
    response = client.get("/api/documents")

    assert response.status_code == 401


def test_documents_endpoint_rejects_invalid_api_key(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "correct-secret")

    response = client.get(
        "/api/documents",
        headers={"X-API-Key": "wrong-secret"},
    )

    assert response.status_code == 401


def test_documents_endpoint_accepts_valid_api_key(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "correct-secret")

    response = client.get(
        "/api/documents",
        headers={"X-API-Key": "correct-secret"},
    )

    assert response.status_code == 200
