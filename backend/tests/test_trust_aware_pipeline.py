"""End-to-end integration tests for the complete Trust-Aware Multi-Agent Pipeline."""

from __future__ import annotations

from typing import Any
import pytest

from app.agents.abstention import AbstentionEngine
from app.agents.critic import CriticAgent
from app.agents.orchestrator import TrustAwareOrchestrator
from app.agents.synthesizer import SynthesizerAgent
from app.agents.verifier import VerifierAgent


class FakeRetriever:
    def __init__(self, responses_by_call: list[list[dict[str, Any]]] | None = None):
        self.responses = list(responses_by_call or [])
        self.call_count = 0

    def retrieve(self, question: str, document_id: str, top_k: int = 5) -> list[dict[str, Any]]:
        self.call_count += 1
        if self.responses:
            idx = min(self.call_count - 1, len(self.responses) - 1)
            return list(self.responses[idx])
        return []


class FakeCritic:
    def __init__(self, evaluations_by_call: list[dict[str, Any]] | None = None):
        self.evaluations = list(evaluations_by_call or [])
        self.call_count = 0

    def evaluate(self, question: str, evidence: list[dict[str, Any]], document_id: str | None = None) -> dict[str, Any]:
        self.call_count += 1
        if self.evaluations:
            idx = min(self.call_count - 1, len(self.evaluations) - 1)
            return self.evaluations[idx]
        return {
            "document_id": document_id,
            "evaluations": [
                {
                    "document_id": document_id,
                    "chunk_id": e.get("chunk_id", "c1"),
                    "relevance": True,
                    "support_status": "supported",
                    "quality_assessment": {"evidence_strength": "high", "source_quality": "high", "potential_outdated": False},
                    "contradiction_status": "none",
                    "explanation": "Valid evidence",
                }
                for e in evidence
            ],
            "contradictions": [],
        }


class FakeSynthesizer:
    def synthesize(self, question: str, evidence: list[dict[str, Any]], document_id: str, **kwargs) -> dict[str, Any]:
        return {
            "answer": "## Answer\nThe Transformer relies on self-attention.",
            "claims": [{"claim_id": "c1", "text": "Transformer relies on self-attention", "cited_chunk_ids": ["c1"]}],
            "sources": [{"filename": "paper.pdf", "page_number": 1}],
            "source_information": [{"filename": "paper.pdf", "page_number": 1}],
            "cited_evidence": evidence,
            "document_id": document_id,
        }


class FakeVerifier:
    def verify(self, claims: list[dict[str, Any]], evidence: list[dict[str, Any]], draft_answer: str, document_id: str | None = None) -> dict[str, Any]:
        return {
            "verified_claims": [
                {
                    "claim_id": "c1",
                    "text": "Transformer relies on self-attention",
                    "status": "VERIFIED",
                    "confidence": 0.98,
                    "supporting_chunk_ids": ["c1"],
                    "reasoning": "Corroborated by chunk c1",
                }
            ],
            "verification_score": 1.0,
            "hallucination_risk": "LOW",
            "hallucination_detected": False,
            "summary": {"total_claims": 1, "verified_claims": 1, "unsupported_claims": 0, "contradicted_claims": 0},
            "verified_answer": draft_answer,
        }


def test_full_pipeline_answers_with_high_trust() -> None:
    chunks = [
        {
            "document_id": "doc-attention",
            "chunk_id": "c1",
            "filename": "paper.pdf",
            "page_number": 1,
            "source": "paper.pdf#page=1",
            "text": "The Transformer is based entirely on attention mechanisms.",
            "score": 0.95,
        }
    ]
    orchestrator = TrustAwareOrchestrator(
        retriever=FakeRetriever([chunks]),
        critic_agent=FakeCritic(),
        synthesizer_agent=FakeSynthesizer(),
        verifier_agent=FakeVerifier(),
        abstention_engine=AbstentionEngine(),
    )

    result = orchestrator.run(
        question="What is Transformer based on?",
        document_id="doc-attention",
    )

    assert result["abstention"] is False
    assert "Transformer relies on self-attention" in result["answer"]
    assert result["trust_score"]["trust_level"] == "HIGH_TRUST"
    assert result["trust_score"]["overall_score"] >= 0.70
    assert result["decision"]["action"] == "DIRECT_ANSWER"
    assert len(result["claims"]) == 1
    assert result["claims"][0]["status"] == "VERIFIED"
    assert result["verification"]["hallucination_risk"] == "LOW"
    assert len(result["agent_trace"]) >= 5


def test_full_pipeline_abstains_on_contradictions() -> None:
    chunks = [
        {"document_id": "doc-contra", "chunk_id": "c1", "text": "Product launched in 2020.", "score": 0.9},
        {"document_id": "doc-contra", "chunk_id": "c2", "text": "Product launched in 2024.", "score": 0.9},
    ]
    contra_eval = {
        "document_id": "doc-contra",
        "evaluations": [
            {"document_id": "doc-contra", "chunk_id": "c1", "relevance": True, "support_status": "contradicted", "quality_assessment": {"evidence_strength": "high", "source_quality": "high"}, "contradiction_status": "contradiction_detected: conflicting dates", "explanation": "conflict"},
            {"document_id": "doc-contra", "chunk_id": "c2", "relevance": True, "support_status": "contradicted", "quality_assessment": {"evidence_strength": "high", "source_quality": "high"}, "contradiction_status": "contradiction_detected: conflicting dates", "explanation": "conflict"},
        ],
        "contradictions": [{"chunk_id_a": "c1", "chunk_id_b": "c2", "explanation": "conflicting launch years"}],
    }

    orchestrator = TrustAwareOrchestrator(
        retriever=FakeRetriever([chunks]),
        critic_agent=FakeCritic([contra_eval]),
        synthesizer_agent=FakeSynthesizer(),
        verifier_agent=FakeVerifier(),
        abstention_engine=AbstentionEngine(),
    )

    result = orchestrator.run(
        question="When was the product launched?",
        document_id="doc-contra",
    )

    assert result["abstention"] is True
    assert result["abstention_reason"] == "contradiction_detected"
    assert result["decision"]["action"] == "ABSTAIN"
    assert "Abstention Notice" in result["answer"]
    assert "conflicting" in result["answer"].lower()


def test_full_pipeline_triggers_retrieve_more_loop() -> None:
    # Pass 1: only 1 weakly relevant chunk (triggers retrieve more)
    # Pass 2: returns 2 strong chunks (reaches high trust)
    pass1_chunks = [{"document_id": "doc-loop", "chunk_id": "c1", "text": "Short mention.", "score": 0.4}]
    pass2_chunks = [
        {"document_id": "doc-loop", "chunk_id": "c1", "text": "Short mention.", "score": 0.4},
        {"document_id": "doc-loop", "chunk_id": "c2", "text": "Detailed factual evidence passage.", "score": 0.9},
    ]

    pass1_critic = {
        "document_id": "doc-loop",
        "evaluations": [
            {"document_id": "doc-loop", "chunk_id": "c1", "relevance": True, "support_status": "supported", "quality_assessment": {"evidence_strength": "low", "source_quality": "medium"}, "contradiction_status": "none", "explanation": "weak"},
        ],
        "contradictions": [],
    }
    pass2_critic = {
        "document_id": "doc-loop",
        "evaluations": [
            {"document_id": "doc-loop", "chunk_id": "c1", "relevance": True, "support_status": "supported", "quality_assessment": {"evidence_strength": "low", "source_quality": "medium"}, "contradiction_status": "none", "explanation": "weak"},
            {"document_id": "doc-loop", "chunk_id": "c2", "relevance": True, "support_status": "supported", "quality_assessment": {"evidence_strength": "high", "source_quality": "high"}, "contradiction_status": "none", "explanation": "strong"},
        ],
        "contradictions": [],
    }

    retriever = FakeRetriever([pass1_chunks, pass2_chunks])
    orchestrator = TrustAwareOrchestrator(
        retriever=retriever,
        critic_agent=FakeCritic([pass1_critic, pass2_critic]),
        synthesizer_agent=FakeSynthesizer(),
        verifier_agent=FakeVerifier(),
        abstention_engine=AbstentionEngine(),
    )

    result = orchestrator.run(
        question="What are the details?",
        document_id="doc-loop",
    )

    # Retriever was called twice (initial + retrieve more loop)
    assert retriever.call_count == 2
    assert any(step["step"] == "RETRIEVE_MORE_LOOP" for step in result["agent_trace"])
    assert result["abstention"] is False
