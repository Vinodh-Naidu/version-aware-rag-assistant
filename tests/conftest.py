import pytest


@pytest.fixture(autouse=True)
def configure_test_api_key(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "test-secret-key")