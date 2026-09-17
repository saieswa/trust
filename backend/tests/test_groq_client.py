"""Tests for evidence-grounded Groq answer generation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.llm.groq_client import FALLBACK_ANSWER, GroqClientError, GroqLLMClient


class FakeCompletions:
    def __init__(self):
        self.request = None

    def create(self, **request):
        self.request = request
        message = SimpleNamespace(
            content="The Transformer uses attention mechanisms to process sequences."
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeGroq:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeCompletions())


def test_answers_real_research_paper_question_from_evidence() -> None:
    fake_client = FakeGroq()
    client = GroqLLMClient(client=fake_client, model="llama-3.3-70b-versatile")
    evidence = [
        {
            "document_id": "attention-paper",
            "chunk_id": "attention-paper::chunk-0001",
            "filename": "attention_is_all_you_need.pdf",
            "page_number": 1,
            "text": "The Transformer is based solely on attention mechanisms.",
            "source": "attention_is_all_you_need.pdf#page=1",
        }
    ]

    result = client.answer("What is the Transformer based solely on?", evidence)

    assert result["answer"].startswith("The Transformer")
    assert result["source_references"] == [
        {
            "source": "attention_is_all_you_need.pdf#page=1",
            "filename": "attention_is_all_you_need.pdf",
            "page_number": 1,
        }
    ]
    assert result["evidence_references"][0]["chunk_id"] == evidence[0]["chunk_id"]
    request = fake_client.chat.completions.request
    assert request["model"] == "llama-3.3-70b-versatile"
    assert "Do not use outside knowledge" in request["messages"][0]["content"]
    for heading in ("## Answer", "## Key Points", "## Explanation", "## Evidence", "## Conclusion"):
        assert heading in request["messages"][0]["content"]
    assert "previous conversation content" in request["messages"][0]["content"]
    assert "attention-paper::chunk-0001" in request["messages"][1]["content"]


def test_no_evidence_returns_fallback_without_calling_groq() -> None:
    fake_client = FakeGroq()
    client = GroqLLMClient(client=fake_client)

    result = client.answer("What is not in this paper?", [])

    assert result == {
        "answer": FALLBACK_ANSWER,
        "source_references": [],
        "evidence_references": [],
    }
    assert fake_client.chat.completions.request is None


def test_missing_api_key_is_reported() -> None:
    client = GroqLLMClient(api_key="", client=None)
    with pytest.raises(GroqClientError, match="GROQ_API_KEY"):
        _ = client.client