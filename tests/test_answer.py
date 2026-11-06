
from types import SimpleNamespace

import pytest

from app import answer


def sample_chunk():
    return {
        "text": (
            "Employees may claim up to $500 "
            "for approved travel expenses."
        ),
        "source": "sample-policy.txt",
        "document_id": "doc1",
        "document_version": 1,
        "chunk_index": 0,
        "start_char": 0,
        "end_char": 65,
        "distance": 0.2,
    }


def mock_ollama_response(content):
    return SimpleNamespace(
        message=SimpleNamespace(content=content)
    )


def setup_valid_retrieval(monkeypatch):
    monkeypatch.setattr(
        answer,
        "search_chunks",
        lambda **kwargs: [sample_chunk()],
    )

    monkeypatch.setattr(
        answer,
        "get_document",
        lambda doc_id: {
            "document_id": doc_id,
            "current_version": 1,
        },
    )


def test_answer_includes_valid_source_citation(monkeypatch):
    setup_valid_retrieval(monkeypatch)

    monkeypatch.setattr(
        answer.ollama_client,
        "chat",
        lambda **kwargs: mock_ollama_response(
            """
            {
                "answer": "The travel limit is $500. [S1]",
                "citation_ids": ["S1"],
                "abstain": false
            }
            """
        ),
    )

    result = answer.generate_answer(
        "What is the travel limit?"
    )

    assert result["abstained"] is False
    assert "$500" in result["answer"]

    assert len(result["citations"]) == 1

    citation = result["citations"][0]

    assert citation["filename"] == "sample-policy.txt"
    assert citation["document_id"] == "doc1"
    assert citation["document_version"] == 1
    assert citation["source_id"] == "S1"


def test_answer_abstains_when_no_evidence(monkeypatch):
    monkeypatch.setattr(
        answer,
        "search_chunks",
        lambda **kwargs: [],
    )

    def unexpected_llm_call(**kwargs):
        raise AssertionError(
            "LLM should not be called without evidence"
        )

    monkeypatch.setattr(
        answer.ollama_client,
        "chat",
        unexpected_llm_call,
    )

    result = answer.generate_answer(
        "What is the leave policy?"
    )

    assert result["abstained"] is True
    assert result["citations"] == []


def test_answer_abstains_when_relevance_is_too_low(monkeypatch):
    chunk = sample_chunk()
    chunk["distance"] = 0.95

    monkeypatch.setattr(
        answer,
        "search_chunks",
        lambda **kwargs: [chunk],
    )

    def unexpected_llm_call(**kwargs):
        raise AssertionError(
            "LLM should not be called for irrelevant evidence"
        )

    monkeypatch.setattr(
        answer.ollama_client,
        "chat",
        unexpected_llm_call,
    )

    result = answer.generate_answer(
        "What is the company's holiday policy?"
    )

    assert result["abstained"] is True
    assert result["citations"] == []


def test_answer_rejects_unknown_citation(monkeypatch):
    setup_valid_retrieval(monkeypatch)

    monkeypatch.setattr(
        answer.ollama_client,
        "chat",
        lambda **kwargs: mock_ollama_response(
            """
            {
                "answer": "The travel limit is $500. [S9]",
                "citation_ids": ["S9"],
                "abstain": false
            }
            """
        ),
    )

    with pytest.raises(answer.AnswerGenerationError):
        answer.generate_answer(
            "What is the travel limit?"
        )


def test_answer_rejects_mismatched_inline_citations(monkeypatch):
    setup_valid_retrieval(monkeypatch)

    monkeypatch.setattr(
        answer.ollama_client,
        "chat",
        lambda **kwargs: mock_ollama_response(
            """
            {
                "answer": "The travel limit is $500. [S1]",
                "citation_ids": [],
                "abstain": false
            }
            """
        ),
    )

    result = answer.generate_answer(
        "What is the travel limit?"
    )

    assert result["abstained"] is True
    assert result["citations"] == []


def test_answer_rejects_invalid_json(monkeypatch):
    setup_valid_retrieval(monkeypatch)

    monkeypatch.setattr(
        answer.ollama_client,
        "chat",
        lambda **kwargs: mock_ollama_response(
            "This is not valid JSON"
        ),
    )

    with pytest.raises(answer.AnswerGenerationError):
        answer.generate_answer(
            "What is the travel limit?"
        )


def test_answer_handles_ollama_unavailable(monkeypatch):
    setup_valid_retrieval(monkeypatch)

    def failing_ollama_call(**kwargs):
        raise ConnectionError("Ollama unavailable")

    monkeypatch.setattr(
        answer.ollama_client,
        "chat",
        failing_ollama_call,
    )

    with pytest.raises(answer.AnswerGenerationError):
        answer.generate_answer(
            "What is the travel limit?"
        )


def test_answer_rejects_stale_document_version(monkeypatch):
    monkeypatch.setattr(
        answer,
        "search_chunks",
        lambda **kwargs: [sample_chunk()],
    )

    # Retrieved version is 1, but the current document is version 2.
    monkeypatch.setattr(
        answer,
        "get_document",
        lambda doc_id: {
            "document_id": doc_id,
            "current_version": 2,
        },
    )

    monkeypatch.setattr(
        answer.ollama_client,
        "chat",
        lambda **kwargs: mock_ollama_response(
            """
            {
                "answer": "The travel limit is $500. [S1]",
                "citation_ids": ["S1"],
                "abstain": false
            }
            """
        ),
    )

    result = answer.generate_answer(
        "What is the travel limit?"
    )

    assert result["abstained"] is True
    assert result["citations"] == []


def test_answer_rejects_empty_question():
    with pytest.raises(ValueError):
        answer.generate_answer("   ")


def test_answer_abstains_when_model_requests_abstention(monkeypatch):
    setup_valid_retrieval(monkeypatch)

    monkeypatch.setattr(
        answer.ollama_client,
        "chat",
        lambda **kwargs: mock_ollama_response(
            """
            {
                "answer": "Insufficient evidence",
                "citation_ids": [],
                "abstain": true
            }
            """
        ),
    )

    result = answer.generate_answer(
        "What is the leave policy?"
    )

    assert result["abstained"] is True
    assert result["citations"] == []
