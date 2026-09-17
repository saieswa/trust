"""Tests for the connection between the Verifier and Synthesizer Agents.

Tests the controlled revision loop:
Retriever → Critic → Trust Score → Decision → Synthesizer → Verifier (→ Revision Loop)
"""

from __future__ import annotations

from typing import Any
import pytest

from app.agents.abstention import AbstentionEngine
from app.agents.orchestrator import TrustAwareOrchestrator
from app.agents.synthesizer import SynthesizerAgent, REVISER_SYSTEM_PROMPT
from app.agents.verifier import VerifierAgent


class MockRetriever:
    def __init__(self, chunks: list[dict[str, Any]]):
        self.chunks = chunks

    def retrieve(self, question: str, document_id: str, top_k: int = 4) -> list[dict[str, Any]]:
        return [c for c in self.chunks if c.get("document_id") == document_id][:top_k]


class MockCritic:
    def evaluate(self, question: str, evidence: list[dict[str, Any]], document_id: str | None = None) -> dict[str, Any]:
        return {
            "document_id": document_id,
            "evaluations": [
                {
                    "document_id": document_id,
                    "chunk_id": e.get("chunk_id"),
                    "relevance": True,
                    "support_status": "supported",
                    "quality_assessment": {"evidence_strength": "high", "source_quality": "high", "potential_outdated": False},
                    "contradiction_status": "none",
                    "explanation": "Valid supporting evidence chunk.",
                }
                for e in evidence
            ],
            "contradictions": [],
        }


class MockSynthesizerWithRevision:
    def __init__(self, initial_answer: str, initial_claims: list[dict[str, Any]], revised_answer: str, revised_claims: list[dict[str, Any]]):
        self.initial_answer = initial_answer
        self.initial_claims = initial_claims
        self.revised_answer = revised_answer
        self.revised_claims = revised_claims
        self.synthesize_call_count = 0
        self.revise_call_count = 0
        self.last_revise_unsupported = None

    def synthesize(self, question: str, evidence: list[dict[str, Any]], document_id: str, **kwargs) -> dict[str, Any]:
        self.synthesize_call_count += 1
        return {
            "answer": self.initial_answer,
            "claims": self.initial_claims,
            "sources": [{"filename": "doc.pdf", "page_number": 1}],
            "cited_evidence": evidence,
            "document_id": document_id,
        }

    def revise(
        self,
        question: str,
        draft_answer: str,
        unsupported_claims: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
        document_id: str,
        **kwargs,
    ) -> dict[str, Any]:
        self.revise_call_count += 1
        self.last_revise_unsupported = unsupported_claims
        return {
            "answer": self.revised_answer,
            "claims": self.revised_claims,
            "sources": [{"filename": "doc.pdf", "page_number": 1}],
            "cited_evidence": evidence,
            "document_id": document_id,
        }


class MockVerifierSequential:
    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = responses
        self.call_count = 0

    def verify(self, **kwargs) -> dict[str, Any]:
        idx = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        return self.responses[idx]


def test_verifier_synthesizer_clean_pipeline_zero_revisions() -> None:
    """When all claims in the initial draft are verified:
    Pipeline completes with revision_count=0 and status=SUPPORTED.
    """
    evidence = [
        {
            "chunk_id": "c1",
            "document_id": "doc-test",
            "filename": "doc.pdf",
            "page_number": 1,
            "source": "doc.pdf#page=1",
            "text": "Binary search is an algorithm that runs in logarithmic time O(log n).",
            "score": 0.92,
        }
    ]

    retriever = MockRetriever(evidence)
    critic = MockCritic()
    synth = MockSynthesizerWithRevision(
        initial_answer="Binary search runs in O(log n) time.",
        initial_claims=[{"claim_id": "claim-1", "claim": "Binary search runs in O(log n) time.", "cited_chunk_ids": ["c1"]}],
        revised_answer="",
        revised_claims=[],
    )
    verifier = MockVerifierSequential([
        {
            "status": "SUPPORTED",
            "verification_score": 1.0,
            "requires_revision": False,
            "revised_trust_score": 0.90,
            "claims": [
                {
                    "claim_id": "claim-1",
                    "claim": "Binary search runs in O(log n) time.",
                    "supported": True,
                    "status": "SUPPORTED",
                    "supporting_chunk_ids": ["c1"],
                    "explanation": "Directly confirmed by chunk c1.",
                }
            ],
            "verified_answer": "Binary search runs in O(log n) time.",
        }
    ])

    orchestrator = TrustAwareOrchestrator(
        retriever=retriever,
        critic_agent=critic,
        synthesizer_agent=synth,
        verifier_agent=verifier,
        abstention_engine=AbstentionEngine(),
    )

    result = orchestrator.run(question="What is the complexity?", document_id="doc-test")

    assert result["revision_count"] == 0
    assert result["verification_status"] == "SUPPORTED"
    assert "final_answer" in result
    assert "claim_level_verification" in result
    assert "supporting_evidence" in result
    assert "trust_score" in result
    assert len(result["claim_level_verification"]) == 1
    assert result["claim_level_verification"][0]["supported"] is True


def test_verifier_synthesizer_controlled_revision_recovery() -> None:
    """When Verifier detects an unsupported claim:
    1. Orchestrator calls Synthesizer.revise() with unsupported claim
    2. Synthesizer revises answer using accepted evidence only
    3. Verifier re-evaluates revised answer
    4. Pipeline completes with revision_count=1 and status=SUPPORTED.
    """
    evidence = [
        {
            "chunk_id": "c1",
            "document_id": "doc-test",
            "filename": "doc.pdf",
            "page_number": 1,
            "source": "doc.pdf#page=1",
            "text": "Binary search divides the search space in half at each step.",
            "score": 0.95,
        }
    ]

    retriever = MockRetriever(evidence)
    critic = MockCritic()
    synth = MockSynthesizerWithRevision(
        initial_answer="Binary search halves the search space. It was invented in ancient Babylon.",
        initial_claims=[
            {"claim_id": "c1", "claim": "Binary search halves the search space.", "cited_chunk_ids": ["c1"]},
            {"claim_id": "c2", "claim": "It was invented in ancient Babylon.", "cited_chunk_ids": []},
        ],
        revised_answer="Binary search halves the search space at each step.",
        revised_claims=[
            {"claim_id": "c1", "claim": "Binary search halves the search space at each step.", "cited_chunk_ids": ["c1"]},
        ],
    )

    # Initial verification: claim 2 is UNSUPPORTED -> requires revision
    # Re-verification: revised claim is SUPPORTED -> passes
    verifier = MockVerifierSequential([
        {
            "status": "UNSUPPORTED",
            "verification_score": 0.5,
            "requires_revision": True,
            "revised_trust_score": 0.40,
            "claims": [
                {
                    "claim_id": "c1",
                    "claim": "Binary search halves the search space.",
                    "supported": True,
                    "status": "SUPPORTED",
                    "supporting_chunk_ids": ["c1"],
                    "explanation": "Supported by chunk c1.",
                },
                {
                    "claim_id": "c2",
                    "claim": "It was invented in ancient Babylon.",
                    "supported": False,
                    "status": "UNSUPPORTED",
                    "supporting_chunk_ids": [],
                    "explanation": "No evidence mentions ancient Babylon.",
                },
            ],
            "verified_answer": "Draft with warning",
        },
        {
            "status": "SUPPORTED",
            "verification_score": 1.0,
            "requires_revision": False,
            "revised_trust_score": 0.88,
            "claims": [
                {
                    "claim_id": "c1",
                    "claim": "Binary search halves the search space at each step.",
                    "supported": True,
                    "status": "SUPPORTED",
                    "supporting_chunk_ids": ["c1"],
                    "explanation": "Supported by chunk c1.",
                }
            ],
            "verified_answer": "Binary search halves the search space at each step.",
        },
    ])

    orchestrator = TrustAwareOrchestrator(
        retriever=retriever,
        critic_agent=critic,
        synthesizer_agent=synth,
        verifier_agent=verifier,
        abstention_engine=AbstentionEngine(),
    )

    result = orchestrator.run(question="How does binary search work?", document_id="doc-test")

    # Verified that 1 controlled revision occurred
    assert synth.revise_call_count == 1
    assert result["revision_count"] == 1
    assert result["verification_status"] == "SUPPORTED"
    assert result["final_answer"] == "Binary search halves the search space at each step."
    assert len(result["claim_level_verification"]) == 1
    assert result["claim_level_verification"][0]["supported"] is True


def test_verifier_synthesizer_reports_failed_verification_and_prevents_infinite_loop() -> None:
    """When the answer remains unsupported after the revision attempt:
    1. Terminates strictly after 1 revision (no infinite loop)
    2. Clearly reports verification_status='FAILED'
    3. Includes verification failure banner in final_answer.
    """
    evidence = [
        {
            "chunk_id": "c1",
            "document_id": "doc-test",
            "filename": "doc.pdf",
            "page_number": 1,
            "source": "doc.pdf#page=1",
            "text": "The document discusses sort routines.",
            "score": 0.90,
        }
    ]

    retriever = MockRetriever(evidence)
    critic = MockCritic()
    synth = MockSynthesizerWithRevision(
        initial_answer="Persistent hallucinated claim.",
        initial_claims=[{"claim_id": "c1", "claim": "Persistent hallucinated claim.", "cited_chunk_ids": []}],
        revised_answer="Still hallucinated claim.",
        revised_claims=[{"claim_id": "c1", "claim": "Still hallucinated claim.", "cited_chunk_ids": []}],
    )

    # Both initial and post-revision verifications fail
    verifier = MockVerifierSequential([
        {
            "status": "UNSUPPORTED",
            "verification_score": 0.0,
            "requires_revision": True,
            "revised_trust_score": 0.35,
            "claims": [
                {
                    "claim_id": "c1",
                    "claim": "Persistent hallucinated claim.",
                    "supported": False,
                    "status": "UNSUPPORTED",
                    "supporting_chunk_ids": [],
                    "explanation": "No evidence found.",
                }
            ],
            "verified_answer": "Persistent hallucinated claim.",
        },
        {
            "status": "UNSUPPORTED",
            "verification_score": 0.0,
            "requires_revision": True,
            "revised_trust_score": 0.35,
            "claims": [
                {
                    "claim_id": "c1",
                    "claim": "Still hallucinated claim.",
                    "supported": False,
                    "status": "UNSUPPORTED",
                    "supporting_chunk_ids": [],
                    "explanation": "Still no evidence found.",
                }
            ],
            "verified_answer": "Still hallucinated claim.",
        },
    ])

    orchestrator = TrustAwareOrchestrator(
        retriever=retriever,
        critic_agent=critic,
        synthesizer_agent=synth,
        verifier_agent=verifier,
        abstention_engine=AbstentionEngine(),
    )

    result = orchestrator.run(question="What is this?", document_id="doc-test")

    # Strict loop bound
    assert result["revision_count"] == 1
    assert synth.revise_call_count == 1
    # Clearly reported failure
    assert result["verification_status"] == "FAILED"
    assert "Verification Failed" in result["final_answer"]
    assert result["verification"]["requires_revision"] is True


def test_reviser_system_prompt_forbids_outside_knowledge() -> None:
    """The system prompt for revision must explicitly forbid outside knowledge."""
    prompt = REVISER_SYSTEM_PROMPT
    assert "100% grounded in the accepted evidence" in prompt
    assert "Do NOT introduce outside, general, or pre-trained knowledge" in prompt
    assert "Do NOT invent facts" in prompt
