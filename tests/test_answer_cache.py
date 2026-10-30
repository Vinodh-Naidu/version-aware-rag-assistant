
from app import answer_cache


def sample_answer():
    return {
        "answer": "The travel limit is $500. [S1]",
        "abstained": False,
        "citations": [
            {
                "source_id": "S1",
                "filename": "sample-policy.txt",
                "document_id": "doc1",
                "document_version": 1,
                "chunk_index": 0,
                "start_char": 0,
                "end_char": 60,
                "excerpt": "Employees may claim up to $500.",
            }
        ],
        "cache_hit": False,
    }


def setup_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(
        answer_cache,
        "CACHE_DB_PATH",
        tmp_path / "test_cache.db",
        )


def test_cache_returns_answer_when_version_matches(tmp_path, monkeypatch):
    setup_cache(tmp_path, monkeypatch)

    monkeypatch.setattr(
        answer_cache,
        "get_document",
        lambda doc_id: {"current_version": 1},
    )

    key = answer_cache.create_cache_key(
        "What is the travel limit?",
        3,
        "llama3.1:8b",
    )

    answer_cache.save_cached_answer(
        key,
        "What is the travel limit?",
        sample_answer(),
    )

    result = answer_cache.get_cached_answer(key)

    assert result is not None
    assert result["cache_hit"] is True
    assert "$500" in result["answer"]


def test_cache_rejects_answer_when_document_version_changes(
        tmp_path, monkeypatch
):
    setup_cache(tmp_path, monkeypatch)

    monkeypatch.setattr(
        answer_cache,
        "get_document",
        lambda doc_id: {"current_version": 2},
    )

    key = answer_cache.create_cache_key(
        "What is the travel limit?",
        3,
        "llama3.1:8b",
    )

    answer_cache.save_cached_answer(
        key,
        "What is the travel limit?",
        sample_answer(),
    )

    result = answer_cache.get_cached_answer(key)

    assert result is None


def test_document_update_invalidates_dependent_answers(
        tmp_path, monkeypatch
):
    setup_cache(tmp_path, monkeypatch)

    key = answer_cache.create_cache_key(
        "What is the travel limit?",
        3,
        "llama3.1:8b",
    )

    answer_cache.save_cached_answer(
        key,
        "What is the travel limit?",
        sample_answer(),
    )

    answer_cache.invalidate_document_cache("doc1")

    monkeypatch.setattr(
        answer_cache,
        "get_document",
        lambda doc_id: {"current_version": 1},
    )

    assert answer_cache.get_cached_answer(key) is None


def test_abstained_answer_is_not_cached(tmp_path, monkeypatch):
    setup_cache(tmp_path, monkeypatch)

    key = answer_cache.create_cache_key(
        "What is the leave policy?",
        3,
        "llama3.1:8b",
    )

    answer_cache.save_cached_answer(
        key,
        "What is the leave policy?",
        {
            "answer": "Insufficient evidence.",
            "abstained": True,
            "citations": [],
        },
    )

    assert answer_cache.get_cached_answer(key) is None
