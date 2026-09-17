"""Unit tests for Synthesizer Agent in Trust-Aware RAG."""

from __future__ import annotations

import json
from types import SimpleNamespace
import pytest

from app.agents.synthesizer import SynthesizerAgent, SYNTHESIZER_SYSTEM_PROMPT


class FakeCompletions:
    def __init__(self, response_json: dict | None = None, response_func=None):
        self.response_json = response_json
        self.response_func = response_func
        self.last_messages = None

    def create(self, **request):
        self.last_messages = request.get("messages", [])
        if self.response_func:
            resp = self.response_func(request)
        else:
            resp = self.response_json or {}
        content = json.dumps(resp)
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeGroq:
    def __init__(self, response_json: dict | None = None, response_func=None):
        self.completions = FakeCompletions(response_json=response_json, response_func=response_func)
        self.chat = SimpleNamespace(completions=self.completions)


def test_synthesizer_system_prompt_enforces_strict_grounding() -> None:
    """Verify system prompt contains all required instructions:
    - Answer only from accepted evidence
    - Do not invent facts
    - Do not fill missing information using outside knowledge
    - Do not cite sources not present in evidence
    - Keep answer readable
    - Clearly indicate uncertainty when evidence is incomplete
    """
    prompt = SYNTHESIZER_SYSTEM_PROMPT
    assert "ONLY from accepted evidence" in prompt
    assert "Do NOT invent facts" in prompt
    assert "outside or pre-trained knowledge" in prompt
    assert "Do NOT cite sources" in prompt
    assert "readable" in prompt
    assert "indicate uncertainty" in prompt


def test_question_answer_present_in_document() -> None:
    """Test question whose answer IS present in the document evidence:
    Produces a grounded answer, cited evidence, source information, and claims.
    """
    fake_response = {
        "answer": "Binary search has a worst-case time complexity of O(log n) because it halves the search interval each step.",
        "claims": [
            {
                "claim_id": "claim-1",
                "text": "Binary search has a time complexity of O(log n).",
                "cited_chunk_ids": ["chunk-bs-1"],
            }
        ],
    }
    fake_client = FakeGroq(response_json=fake_response)
    agent = SynthesizerAgent(llm_client=fake_client)

    evidence = [
        {
            "chunk_id": "chunk-bs-1",
            "document_id": "doc-algo",
            "filename": "algorithms.pdf",
            "page_number": 4,
            "source": "algorithms.pdf#page=4",
            "text": "Binary search operates in O(log n) time complexity by halving the search space at each iteration.",
        }
    ]

    result = agent.synthesize(
        question="What is the time complexity of binary search?",
        evidence=evidence,
        document_id="doc-algo",
        metadata={"filename": "algorithms.pdf", "pages": 10},
        trust_decision={"decision": "CONTINUE_TO_SYNTHESIZER", "trust_score": 0.88},
    )

    # 1. Output structure
    assert "answer" in result
    assert "cited_evidence" in result
    assert "source_information" in result
    assert "claims" in result

    # 2. Grounded answer and claims
    assert "O(log n)" in result["answer"]
    assert len(result["claims"]) == 1
    assert result["claims"][0]["claim_id"] == "claim-1"
    assert result["claims"][0]["cited_chunk_ids"] == ["chunk-bs-1"]

    # 3. Cited evidence & source information
    assert len(result["cited_evidence"]) == 1
    assert result["cited_evidence"][0]["chunk_id"] == "chunk-bs-1"
    assert result["cited_evidence"][0]["document_id"] == "doc-algo"
    assert len(result["source_information"]) == 1
    assert result["source_information"][0]["filename"] == "algorithms.pdf"
    assert result["source_information"][0]["page_number"] == 4

    # 4. Trust decision and metadata preserved
    assert result["trust_decision"]["decision"] == "CONTINUE_TO_SYNTHESIZER"
    assert result["metadata"]["filename"] == "algorithms.pdf"


def test_question_answer_not_present_in_document() -> None:
    """Test question whose answer is NOT present in the document:
    Must NOT invent facts using outside knowledge, and must indicate uncertainty/absence of evidence.
    """
    fake_response = {
        "answer": "Based on the provided document evidence, there is no information to answer this question. The document covers binary search algorithms and does not mention quantum encryption protocols.",
        "claims": [],
    }
    fake_client = FakeGroq(response_json=fake_response)
    agent = SynthesizerAgent(llm_client=fake_client)

    evidence = [
        {
            "chunk_id": "chunk-bs-1",
            "document_id": "doc-algo",
            "filename": "algorithms.pdf",
            "page_number": 4,
            "source": "algorithms.pdf#page=4",
            "text": "Binary search works on sorted arrays by repeatedly dividing the search interval in half.",
        }
    ]

    result = agent.synthesize(
        question="What is the quantum encryption protocol used in this system?",
        evidence=evidence,
        document_id="doc-algo",
    )

    # Must clearly indicate absence of evidence without inventing facts
    assert "no information to answer this question" in result["answer"].lower() or "does not mention" in result["answer"].lower()
    # No false claims invented
    assert result["claims"] == []
    assert result["document_id"] == "doc-algo"


def test_synthesizer_excludes_rejected_chunks() -> None:
    """The Synthesizer must NOT receive or cite rejected chunks from the Critic."""
    fake_client = FakeGroq(response_func=lambda req: {
        "answer": "Ground truth from approved chunk.",
        "claims": [{"claim_id": "c1", "text": "fact", "cited_chunk_ids": ["chunk-approved"]}],
    })
    agent = SynthesizerAgent(llm_client=fake_client)

    evidence = [
        {
            "chunk_id": "chunk-approved",
            "document_id": "doc-1",
            "text": "Valid approved text.",
        },
        {
            "chunk_id": "chunk-rejected-by-flag",
            "document_id": "doc-1",
            "text": "Rejected irrelevant text.",
            "is_rejected": True,
        },
        {
            "chunk_id": "chunk-rejected-by-eval",
            "document_id": "doc-1",
            "text": "Evaluated as irrelevant.",
        },
    ]

    evaluations = [
        {"chunk_id": "chunk-approved", "relevance": True, "support_status": "supported"},
        {"chunk_id": "chunk-rejected-by-eval", "relevance": False, "support_status": "rejected"},
    ]

    result = agent.synthesize(
        question="What is the fact?",
        evidence=evidence,
        document_id="doc-1",
        evaluations=evaluations,
    )

    # Inspect the prompt that was sent to the LLM
    last_prompt = fake_client.completions.last_messages[1]["content"]
    assert "chunk-approved" in last_prompt
    assert "chunk-rejected-by-flag" not in last_prompt
    assert "chunk-rejected-by-eval" not in last_prompt

    # Only approved chunk in cited evidence
    assert len(result["cited_evidence"]) == 1
    assert result["cited_evidence"][0]["chunk_id"] == "chunk-approved"


def test_synthesizer_excludes_unrelated_documents() -> None:
    """The Synthesizer must NOT receive chunks belonging to unrelated documents."""
    fake_client = FakeGroq(response_func=lambda req: {
        "answer": "Answer for doc-target.",
        "claims": [{"claim_id": "c1", "text": "target fact", "cited_chunk_ids": ["c-target"]}],
    })
    agent = SynthesizerAgent(llm_client=fake_client)

    evidence = [
        {
            "chunk_id": "c-target",
            "document_id": "doc-target",
            "text": "Legitimate text from target document.",
        },
        {
            "chunk_id": "c-leaked",
            "document_id": "doc-unrelated",
            "text": "Leaked text from unrelated document.",
        },
        {
            "chunk_id": "c-unrelated-tag",
            "document_id": "doc-target",
            "text": "Tagged as unrelated by critic.",
            "support_status": "unrelated_document",
        },
    ]

    result = agent.synthesize(
        question="What is the target info?",
        evidence=evidence,
        document_id="doc-target",
    )

    last_prompt = fake_client.completions.last_messages[1]["content"]
    assert "c-target" in last_prompt
    assert "c-leaked" not in last_prompt
    assert "c-unrelated-tag" not in last_prompt


def test_synthesizer_excludes_contradictory_evidence_unless_explicitly_marked() -> None:
    """The Synthesizer must NOT receive contradictory evidence unless explicitly marked."""
    fake_client = FakeGroq(response_func=lambda req: {
        "answer": "Answer addressing marked contradiction.",
        "claims": [{"claim_id": "c1", "text": "fact", "cited_chunk_ids": ["chunk-marked-contra"]}],
    })
    agent = SynthesizerAgent(llm_client=fake_client)

    evidence = [
        {
            "chunk_id": "chunk-clean",
            "document_id": "doc-test",
            "text": "Clean uncontradicted evidence.",
        },
        {
            "chunk_id": "chunk-unmarked-contra",
            "document_id": "doc-test",
            "text": "Unmarked contradictory claim stating accuracy is 40%.",
        },
        {
            "chunk_id": "chunk-marked-contra",
            "document_id": "doc-test",
            "text": "Marked contradictory claim stating accuracy is 90%.",
            "has_contradiction": True,
            "contradiction_reason": "Numerical conflict with baseline report.",
        },
    ]

    contradictions = [
        {
            "chunk_a": "chunk-unmarked-contra",
            "chunk_b": "other-chunk",
            "reason": "Numerical conflict",
        }
    ]

    agent.synthesize(
        question="What is the accuracy?",
        evidence=evidence,
        document_id="doc-test",
        contradictions=contradictions,
    )

    last_prompt = fake_client.completions.last_messages[1]["content"]
    # Clean chunk is included
    assert "chunk-clean" in last_prompt
    # Unmarked contradictory chunk is EXCLUDED
    assert "chunk-unmarked-contra" not in last_prompt
    # Explicitly marked contradictory chunk is INCLUDED with warning notice
    assert "chunk-marked-contra" in last_prompt
    assert "CONTRADICTION DETECTED" in last_prompt


def test_synthesizer_empty_evidence_returns_uncertainty() -> None:
    """When evidence is empty or all filtered out, return abstention notice without LLM call."""
    agent = SynthesizerAgent(llm_client=FakeGroq({}))
    result = agent.synthesize(
        question="What is the answer?",
        evidence=[],
        document_id="doc-1",
    )

    assert "don't have enough reliable evidence" in result["answer"].lower()
    assert result["cited_evidence"] == []
    assert result["source_information"] == []
    assert result["claims"] == []
