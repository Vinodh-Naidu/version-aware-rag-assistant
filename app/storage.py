
import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from app.ingestion import ProcessedDocument


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STORE_PATH = DATA_DIR / "documents.json"

_store_lock = threading.RLock()


def _load_store() -> dict:
    if not STORE_PATH.exists():
        return {"documents": {}}

    with STORE_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_store(store: dict) -> None:
    """Atomically replace the JSON file after a complete write."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    fd, temporary_path = tempfile.mkstemp(
        dir=DATA_DIR,
        prefix="documents-",
        suffix=".tmp",
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(store, file, indent=2, ensure_ascii=False)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_path, STORE_PATH)

    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name)
    name = name.strip(" .")

    if not name or name in {".", ".."}:
        raise ValueError("Invalid filename.")

    return name[:150]


def _document_id(filename: str) -> str:
    # Same filename identifies the same logical document in this demo.
    identity = filename.casefold().encode("utf-8")
    return sha256(identity).hexdigest()[:20]


def upsert_document(document: ProcessedDocument) -> dict:
    filename = _safe_filename(document.filename)
    doc_id = _document_id(filename)

    with _store_lock:
        store = _load_store()
        existing = store["documents"].get(doc_id)

        if existing and existing["filename"] != filename:
            raise ValueError(
                "Filename identity collision. Choose another filename."
            )

        if (
                existing
                and existing["versions"][-1]["content_hash"]
                == document.content_hash
        ):
            current = existing["versions"][-1]

            return {
                "document_id": doc_id,
                "filename": filename,
                "version": current["version"],
                "content_hash": current["content_hash"],
                "chunk_count": len(current["chunks"]),
                "status": "unchanged",
            }

        version_number = (
            existing["versions"][-1]["version"] + 1
            if existing
            else 1
        )

        version = {
            "version": version_number,
            "content_hash": document.content_hash,
            "text": document.text,
            "chunks": [
                {
                    **chunk,
                    "chunk_id": (
                        f"{doc_id}:v{version_number}:"
                        f"{chunk['chunk_index']}"
                    ),
                    "document_id": doc_id,
                    "document_version": version_number,
                    "filename": filename,
                }
                for chunk in document.chunks
            ],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        if existing:
            existing["versions"].append(version)
        else:
            store["documents"][doc_id] = {
                "filename": filename,
                "versions": [version],
            }

        _save_store(store)

        return {
            "document_id": doc_id,
            "filename": filename,
            "version": version_number,
            "content_hash": document.content_hash,
            "chunk_count": len(document.chunks),
            "status": "created" if version_number == 1 else "updated",
        }


def list_documents() -> list[dict]:
    with _store_lock:
        store = _load_store()

        result = []

        for doc_id, document in store["documents"].items():
            current = document["versions"][-1]

            result.append({
                "document_id": doc_id,
                "filename": document["filename"],
                "current_version": current["version"],
                "content_hash": current["content_hash"],
                "chunk_count": len(current["chunks"]),
                "created_at": current["created_at"],
            })

        return result


def get_document(doc_id: str) -> dict | None:
    with _store_lock:
        store = _load_store()
        document = store["documents"].get(doc_id)

        if document is None:
            return None

        current = document["versions"][-1]

        return {
            "document_id": doc_id,
            "filename": document["filename"],
            "current_version": current["version"],
            "content_hash": current["content_hash"],
            "text": current["text"],
            "chunks": current["chunks"],
            "history": [
                {
                    "version": item["version"],
                    "content_hash": item["content_hash"],
                    "chunk_count": len(item["chunks"]),
                    "created_at": item["created_at"],
                }
                for item in document["versions"]
            ],
        }
