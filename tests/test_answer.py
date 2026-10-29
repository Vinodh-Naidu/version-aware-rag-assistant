
from types import SimpleNamespace

from app import answer


def sample_chunk():
    return {
        "text": "Employees may claim up to $500 for approved travel expenses.",
        "source": "sample-policy.txt",
        "document_id": "doc1",
        "document_version": 1,
        "chunk_index": 0,
        "start_char": 0,
        "end_char": 60,
        "distance": 0.2,
    }


def test_answer_includes_valid_source_citation(monkeypatch):
    monkeypatch.setattr(
        answer, "search_chunks", lambda **kwargs: [sample_chunk()]
    )
    monkeypatch.setattr(
        answer.ollama,
        "chat",
        lambda **kwargs: SimpleNamespace(
            message=SimpleNamespace(
                content=(
                    '{"answer":"The limit is $500. [S1]",'
                    '"citation_ids":["S1"],"abstain":false}'
                )
            )
        ),
    )

    result = answer.generate_answer("What is the travel limit?")

    assert result["abstained"] is False
    assert "$500" in result["answer"]
    assert result["citations"][0]["filename"] == "sample-policy.txt"
    assert result["citations"][0]["document_version"] == 1


def test_answer_abstains_when_no_evidence(monkeypatch):
    monkeypatch.setattr(answer, "search_chunks", lambda **kwargs: [])

    def unexpected_llm_call(**kwargs):
        raise AssertionError("LLM should not be called without evidence")

    monkeypatch.setattr(answer.ollama, "chat", unexpected_llm_call)

    result = answer.generate_answer("What is the leave policy?")

    assert result["abstained"] is True
    assert result["citations"] == []


def test_answer_rejects_unknown_citation(monkeypatch):
    monkeypatch.setattr(
        answer, "search_chunks", lambda **kwargs: [sample_chunk()]
    )
    monkeypatch.setattr(
        answer.ollama,
        "chat",
        lambda **kwargs: SimpleNamespace(
            message=SimpleNamespace(
                content=(
                    '{"answer":"The limit is $500. [S9]",'
                    '"citation_ids":["S9"],"abstain":false}'
                )
            )
        ),
    )

    try:
        answer.generate_answer("What is the travel limit?")
        assert False, "Expected invalid citation to be rejected"
    except answer.AnswerGenerationError:
        pass
