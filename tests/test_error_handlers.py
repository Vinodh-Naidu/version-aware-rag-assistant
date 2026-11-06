
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_unknown_endpoint_returns_structured_404():
    response = client.get("/api/does-not-exist")

    assert response.status_code == 404

    body = response.json()

    assert body["error"]["code"] == "NOT_FOUND"
    assert body["error"]["message"] == (
        "The requested resource was not found."
    )
    assert body["error"]["request_id"] != "unknown"

    assert response.headers.get("X-Request-ID") == (
        body["error"]["request_id"]
    )


def test_validation_error_returns_structured_response():
    response = client.post(
        "/api/retrieve",
        json={},
        headers={"X-API-Key": "test-secret-key"},
    )

    assert response.status_code == 422

    body = response.json()
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert body["error"]["request_id"]