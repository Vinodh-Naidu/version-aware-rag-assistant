
import io

import pytest
from fastapi.testclient import TestClient

from app.ingestion import (
    DocumentError,
    chunk_text,
    extract_text,
)
from app.main import app
from app import storage


client = TestClient(app)


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        storage,
        "STORE_PATH",
        tmp_path / "documents.json",
        )


def upload_text(filename: str, text: str):
    return client.post(
        "/api/documents/upload",
        files={
            "file": (
                filename,
                text.encode("utf-8"),
                "text/plain",
            )
        },
    )


def test_upload_creates_version_one(isolated_store):
    response = upload_text(
        "policy.txt",
        "Lodging (Standard): $250 per night.",
    )

    assert response.status_code == 201
    assert response.json()["version"] == 1
    assert response.json()["status"] == "created"


def test_identical_upload_does_not_create_new_version(
        isolated_store,
):
    first = upload_text("policy.txt", "Lodging $250 per night.")
    second = upload_text("policy.txt", "Lodging $250 per night.")

    assert first.json()["document_id"] == second.json()["document_id"]
    assert second.json()["version"] == 1
    assert second.json()["status"] == "unchanged"


def test_changed_content_increments_version(isolated_store):
    upload_text("policy.txt", "Lodging $250 per night.")

    response = upload_text("policy.txt", "Lodging $300 per night.")

    assert response.status_code == 201
    assert response.json()["version"] == 2
    assert response.json()["status"] == "updated"


def test_changed_content_has_new_hash(isolated_store):
    first = upload_text("policy.txt", "Lodging $250 per night.")
    second = upload_text("policy.txt", "Lodging $300 per night.")

    assert (
            first.json()["content_hash"]
            != second.json()["content_hash"]
    )


def test_unsupported_extension_returns_400(isolated_store):
    response = upload_text("policy.exe", "not a document")

    assert response.status_code == 400


def test_empty_text_returns_400(isolated_store):
    response = upload_text("empty.txt", "   \n ")

    assert response.status_code == 400


def test_chunk_overlap():
    text = "A" * 2500

    chunks = chunk_text(text, chunk_size=1000, overlap=100)

    assert len(chunks) == 3
    assert chunks[0]["text"][-100:] == chunks[1]["text"][:100]
    assert chunks[1]["text"][-100:] == chunks[2]["text"][:100]


def test_invalid_chunk_configuration():
    with pytest.raises(ValueError):
        chunk_text("example", chunk_size=100, overlap=100)


