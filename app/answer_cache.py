import hashlib
import json
import sqlite3
from pathlib import Path

from app.storage import get_document

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DB_PATH = BASE_DIR / "data" / "cache.db"


def get_db_connection():
    """Establish a SQLite connection and ensure the answer_cache table exists."""
    CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CACHE_DB_PATH))
    conn.row_factory = sqlite3.Row

    conn.execute("""
        CREATE TABLE IF NOT EXISTS answer_cache (
            cache_key TEXT PRIMARY KEY,
            question TEXT,
            answer_json TEXT,
            document_versions TEXT
        )
    """)
    conn.commit()
    return conn


def create_cache_key(question: str, top_k: int, model_name: str) -> str:
    """Generate a deterministic SHA-256 hash for the query configuration."""
    raw = f"{question.strip()}:{top_k}:{model_name}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def save_cached_answer(cache_key: str, question: str, answer: dict):
    """Save an answer to the cache, unless it is an abstained response."""
    if answer.get("abstained"):
        return

    # Extract document versions from citations for version tracking
    doc_versions = {}
    for citation in answer.get("citations", []):
        doc_id = citation.get("document_id")
        doc_version = citation.get("document_version")
        if doc_id:
            doc_versions[doc_id] = doc_version

    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO answer_cache 
            (cache_key, question, answer_json, document_versions)
            VALUES (?, ?, ?, ?)
            """,
            (
                cache_key,
                question,
                json.dumps(answer),
                json.dumps(doc_versions),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_cached_answer(cache_key: str) -> dict | None:
    """Retrieve a cached answer if it exists and underlying document versions match."""
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            "SELECT answer_json, document_versions FROM answer_cache WHERE cache_key = ?",
            (cache_key,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        answer = json.loads(row["answer_json"])
        doc_versions = json.loads(row["document_versions"])

        # Validate that dependent documents haven't changed versions
        for doc_id, cached_version in doc_versions.items():
            current_doc = get_document(doc_id)
            if not current_doc or current_doc.get("current_version") != cached_version:
                conn.execute(
                    "DELETE FROM answer_cache WHERE cache_key = ?",
                    (cache_key,),
                )
                conn.commit()
                return None

        answer["cache_hit"] = True
        return answer
    finally:
        conn.close()


def invalidate_document_cache(document_id: str):
    """Invalidate all cached answers dependent on a specific document ID."""
    conn = get_db_connection()
    try:
        cursor = conn.execute("SELECT cache_key, document_versions FROM answer_cache")
        for row in cursor.fetchall():
            doc_versions = json.loads(row["document_versions"])
            if document_id in doc_versions:
                conn.execute(
                    "DELETE FROM answer_cache WHERE cache_key = ?",
                    (row["cache_key"],),
                )
        conn.commit()
    finally:
        conn.close()