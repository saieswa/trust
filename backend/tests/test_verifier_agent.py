"""Unit tests for Verifier Agent and Claim-Level Verification."""

from __future__ import annotations

import json
from types import SimpleNamespace
import pytest

from app.agents.verifier import VerifierAgent


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


def test_case_1_fully_supported_answer() -> None:
    """Case 1: Fully supported answer.
    All claims are verified against the accepted evidence.
    Returns: status=SUPPORTED, supported=True, requires_revision=False.
    """
    fake_response = {
        "claim_evaluations": [
            {
                "claim_id": "c1",
                "claim": "Binary search has a time complexity of O(log n).",
                "status": "SUPPORTED",
                "supported": True,
                "supporting_chunk_ids": ["chunk-1"],
                "explanation": "Directly confirmed by chunk-1 which states binary search runs in O(log n) time.",
            }
        ]
    }
    agent = VerifierAgent(llm_client=FakeGroq(fake_response))

    question = "What is the time complexity of binary search?"
    answer = "Binary search has a time complexity of O(log n)."
    accepted_evidence = [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-algo",
            "text": "Binary search runs in O(log n) time complexity on sorted arrays.",
        }
    ]

    result = agent.verify(
        question=question,
        generated_answer=answer,
        accepted_evidence=accepted_evidence,
        document_id="doc-algo",
        trust_score=0.85,
    )

    assert result["status"] == "SUPPORTED"
    assert result["verification_score"] == 1.0
    assert result["requires_revision"] is False
    assert result["hallucination_detected"] is False
    assert result["hallucination_risk"] == "LOW"
    assert result["revised_trust_score"] == 0.85

    # Check claim structure
    assert len(result["claims"]) == 1
    c = result["claims"][0]
    assert c["claim"] == "Binary search has a time complexity of O(log n)."
    assert c["supported"] is True
    assert c["status"] == "SUPPORTED"
    assert c["supporting_chunk_ids"] == ["chunk-1"]
    assert len(c["explanation"]) > 0


def test_case_2_partially_supported_answer() -> None:
    """Case 2: Partially supported answer.
    Contains claims with status PARTIALLY_SUPPORTED.
    Must mark answer as requiring revision and adjust trust score.
    """
    fake_response = {
        "claim_evaluations": [
            {
                "claim_id": "c1",
                "claim": "Binary search runs in O(log n) time and was created by Ada Lovelace in 1843.",
                "status": "PARTIALLY_SUPPORTED",
                "supported": False,
                "supporting_chunk_ids": ["chunk-1"],
                "explanation": "Evidence supports O(log n) complexity, but does not mention Ada Lovelace.",
            }
        ]
    }
    agent = VerifierAgent(llm_client=FakeGroq(fake_response))

    result = agent.verify(
        question="Who invented binary search and what is its complexity?",
        generated_answer="Binary search runs in O(log n) time and was created by Ada Lovelace in 1843.",
        accepted_evidence=[
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-algo",
                "text": "Binary search operates in O(log n) time complexity.",
            }
        ],
        document_id="doc-algo",
        trust_score=0.80,
    )

    assert result["status"] == "PARTIALLY_SUPPORTED"
    assert result["requires_revision"] is True
    assert result["hallucination_detected"] is True
    assert result["revised_trust_score"] < 0.80

    c = result["claims"][0]
    assert c["status"] == "PARTIALLY_SUPPORTED"
    assert c["supported"] is False
    assert "chunk-1" in c["supporting_chunk_ids"]
    assert "does not mention Ada Lovelace" in c["explanation"]

    # Verify not silently presented as verified
    assert "Revision Required" in result["verified_answer"]


def test_case_3_hallucinated_claim() -> None:
    """Case 3: Hallucinated claim.
    The answer invents unstated facts not in the evidence.
    Returns: status=UNSUPPORTED, supported=False, requires_revision=True.
    """
    fake_response = {
        "claim_evaluations": [
            {
                "claim_id": "c1",
                "claim": "Binary search utilizes quantum teleportation protocols.",
                "status": "UNSUPPORTED",
                "supported": False,
                "supporting_chunk_ids": [],
                "explanation": "Evidence mentions sorted arrays and has no mention of quantum teleportation.",
            }
        ]
    }
    agent = VerifierAgent(llm_client=FakeGroq(fake_response))

    result = agent.verify(
        question="How does binary search work?",
        generated_answer="Binary search utilizes quantum teleportation protocols.",
        accepted_evidence=[
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-algo",
                "text": "Binary search operates on sorted arrays by dividing the range.",
            }
        ],
        document_id="doc-algo",
        trust_score=0.85,
    )

    assert result["status"] == "UNSUPPORTED"
    assert result["requires_revision"] is True
    assert result["hallucination_detected"] is True
    assert result["hallucination_risk"] == "HIGH"
    assert result["revised_trust_score"] <= 0.40

    c = result["claims"][0]
    assert c["status"] == "UNSUPPORTED"
    assert c["supported"] is False
    assert c["supporting_chunk_ids"] == []
    assert "no mention of quantum" in c["explanation"].lower()


def test_case_4_incorrect_claim() -> None:
    """Case 4: Incorrect claim.
    Direct factual or numerical contradiction with the evidence.
    Returns: status=UNSUPPORTED, supported=False.
    """
    fake_response = {
        "claim_evaluations": [
            {
                "claim_id": "c1",
                "claim": "The model achieved 95% accuracy.",
                "status": "UNSUPPORTED",
                "supported": False,
                "supporting_chunk_ids": [],
                "explanation": "Incorrect claim: Evidence states accuracy was 72%, not 95%.",
            }
        ]
    }
    agent = VerifierAgent(llm_client=FakeGroq(fake_response))

    result = agent.verify(
        question="What was the accuracy?",
        generated_answer="The model achieved 95% accuracy.",
        accepted_evidence=[
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-report",
                "text": "The model achieved 72% accuracy on the evaluation benchmark.",
            }
        ],
        document_id="doc-report",
        trust_score=0.78,
    )

    assert result["status"] == "UNSUPPORTED"
    assert result["requires_revision"] is True
    assert result["claims"][0]["supported"] is False
    assert result["claims"][0]["status"] == "UNSUPPORTED"


def test_case_5_claim_supported_by_another_document_but_not_current_document() -> None:
    """Case 5: Claim supported by another document, but NOT the current document.
    The Verifier must NOT use external knowledge as evidence or allow cross-document support.
    Must return: status=UNSUPPORTED, supported=False.
    """
    agent = VerifierAgent(llm_client=FakeGroq({}))

    # Current document is "doc-binary-search"
    # Claim is about QuickSort, which belongs to a leaked chunk from "doc-quicksort"
    accepted_evidence = [
        {
            "chunk_id": "chunk-bs",
            "document_id": "doc-binary-search",
            "text": "Binary search is an algorithm for finding an element in a sorted list in logarithmic time.",
        },
        {
            "chunk_id": "chunk-qs-leaked",
            "document_id": "doc-quicksort",
            "text": "QuickSort has an average time complexity of O(n log n).",
        },
    ]

    result = agent.verify(
        question="What is the complexity of QuickSort?",
        generated_answer="QuickSort has an average time complexity of O(n log n).",
        accepted_evidence=accepted_evidence,
        document_id="doc-binary-search",
        trust_score=0.82,
    )

    # Cross-document chunk must NOT support the claim in doc-binary-search
    assert result["status"] == "UNSUPPORTED"
    assert result["requires_revision"] is True
    assert result["claims"][0]["supported"] is False
    assert result["claims"][0]["status"] == "UNSUPPORTED"
    assert "chunk-qs-leaked" not in result["claims"][0]["supporting_chunk_ids"]
    assert "cross-document" in result["claims"][0]["explanation"].lower() or "not substantiated" in result["claims"][0]["explanation"].lower()


def test_verifier_handles_empty_claims() -> None:
    """Verifier handles empty claims gracefully."""
    agent = VerifierAgent(llm_client=FakeGroq({}))
    result = agent.verify(claims=[], evidence=[], draft_answer="Hello")

    assert result["status"] == "SUPPORTED"
    assert result["verification_score"] == 1.0
    assert result["requires_revision"] is False
    assert result["claims"] == []
