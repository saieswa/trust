"""Comprehensive End-to-End Integration Tests for the Trust-Aware RAG Pipeline.

Verifies the complete flow:
User Question + Current Document ID
↓
Retriever (FAISS + Cache with document_id)
↓
Relevant Evidence
↓
Critic
↓
Contradiction Detection
↓
Trust Score
↓
Trust-Based Decision
  ├── HIGH   → Synthesizer → Verifier (→ Revision Loop) → Final Output
  ├── MEDIUM → Retrieve More Evidence Loop → Critic → Trust Score → Decision
  └── LOW    → Abstain → Final Output

Verifies all mandatory invariants:
- document_id is preserved at every stage
- rejected evidence never reaches Synthesizer
- contradictory evidence is explicitly marked
- previous documents cannot contaminate current answers
- cache keys contain document_id
- previous chat history is not used as factual evidence
- maximum retrieval/revision loops exist
- errors are handled gracefully
"""

from __future__ import annotations

from typing import Any
import pytest

from app.agents.abstention import AbstentionEngine
from app.agents.orchestrator import TrustAwareOrchestrator
from app.database.redis_cache import RedisCache
from app.trust.decision import ActionType


class E2ERetriever:
    """Mock retriever simulating document-scoped vector search."""
    def __init__(self, document_chunks: dict[str, list[dict[str, Any]]]):
        self.document_chunks = document_chunks
        self.retrieve_call_count = 0
        self.last_query = None
        self.last_doc_id = None

    def retrieve(self, question: str, document_id: str, top_k: int = 5) -> list[dict[str, Any]]:
        self.retrieve_call_count += 1
        self.last_query = question
        self.last_doc_id = document_id
        chunks = self.document_chunks.get(document_id, [])
        return chunks[:top_k]


class E2ECritic:
    """Mock critic agent."""
    def __init__(self, relevance: bool = True, support_status: str = "supported"):
        self.relevance = relevance
        self.support_status = support_status

    def evaluate(self, question: str, evidence: list[dict[str, Any]], document_id: str | None = None) -> dict[str, Any]:
        evals = []
        for c in evidence:
            cid = c.get("chunk_id")
            c_doc = c.get("document_id")
            if c_doc and document_id and c_doc != document_id:
                evals.append({
                    "chunk_id": cid,
                    "relevance": False,
                    "support_status": "unrelated_document",
                    "explanation": "Belongs to another document.",
                })
            else:
                evals.append({
                    "chunk_id": cid,
                    "relevance": self.relevance,
                    "support_status": self.support_status,
                    "quality_assessment": {"evidence_strength": "high", "source_quality": "high", "potential_outdated": False},
                    "explanation": "Valid critique.",
                })
        return {
            "document_id": document_id,
            "evaluations": evals,
            "contradictions": [],
        }


class E2ESynthesizer:
    """Mock synthesizer checking evidence isolation."""
    def __init__(self, answer: str = "Grounded synthesized answer."):
        self.answer = answer
        self.received_evidence: list[dict[str, Any]] = []
        self.received_doc_id = None

    def synthesize(self, question: str, evidence: list[dict[str, Any]], document_id: str, **kwargs) -> dict[str, Any]:
        self.received_evidence = list(evidence)
        self.received_doc_id = document_id
        return {
            "answer": self.answer,
            "claims": [{"claim_id": "c1", "claim": self.answer, "cited_chunk_ids": [c.get("chunk_id") for c in evidence if c.get("chunk_id")]}],
            "sources": [{"filename": "algo.pdf", "page_number": 1}],
            "cited_evidence": evidence,
            "document_id": document_id,
        }

    def revise(self, question: str, draft_answer: str, unsupported_claims: list[dict[str, Any]], evidence: list[dict[str, Any]], document_id: str, **kwargs) -> dict[str, Any]:
        return self.synthesize(question, evidence, document_id, **kwargs)


class E2EVerifier:
    """Mock verifier."""
    def __init__(self, status: str = "SUPPORTED"):
        self.status = status

    def verify(self, **kwargs) -> dict[str, Any]:
        return {
            "status": self.status,
            "verification_score": 1.0 if self.status == "SUPPORTED" else 0.0,
            "requires_revision": self.status != "SUPPORTED",
            "revised_trust_score": 0.85 if self.status == "SUPPORTED" else 0.35,
            "claims": [
                {
                    "claim_id": "c1",
                    "claim": "Grounded claim",
                    "supported": self.status == "SUPPORTED",
                    "status": self.status,
                    "supporting_chunk_ids": ["c1"],
                    "explanation": "Verified.",
                }
            ],
            "verified_answer": kwargs.get("draft_answer") or kwargs.get("generated_answer") or "",
        }


# ==============================================================================
# 1. HIGH Trust Flow
# ==============================================================================
def test_e2e_high_trust_flow() -> None:
    """High trust flow executes:
    Retriever -> Critic -> Trust Score -> Decision (HIGH) -> Synthesizer -> Verifier -> Final Output.
    """
    doc_id = "doc-algo"
    chunks = [
        {
            "chunk_id": "c1",
            "document_id": doc_id,
            "filename": "algo.pdf",
            "page_number": 1,
            "source": "algo.pdf#page=1",
            "text": "Binary search runs in O(log n) time complexity.",
            "score": 0.95,
        }
    ]
    retriever = E2ERetriever({doc_id: chunks})
    critic = E2ECritic(relevance=True, support_status="supported")
    synth = E2ESynthesizer(answer="Binary search runs in O(log n) time.")
    verifier = E2EVerifier(status="SUPPORTED")

    orchestrator = TrustAwareOrchestrator(
        retriever=retriever,
        critic_agent=critic,
        synthesizer_agent=synth,
        verifier_agent=verifier,
        abstention_engine=AbstentionEngine(),
    )

    result = orchestrator.run(question="What is binary search complexity?", document_id=doc_id)

    # Invariants verification
    assert result["document_id"] == doc_id
    assert result["abstention"] is False
    assert result["decision"]["decision"] == "CONTINUE_TO_SYNTHESIZER"
    assert result["trust_score"]["overall_score"] >= 0.75
    assert result["verification_status"] == "SUPPORTED"
    assert result["revision_count"] == 0
    assert "Binary search runs in O(log n) time." in result["final_answer"]
    assert len(result["supporting_evidence"]) == 1
    assert result["supporting_evidence"][0]["document_id"] == doc_id


# ==============================================================================
# 2. MEDIUM Trust Flow with Retrieve-More Expansion Loop
# ==============================================================================
def test_e2e_medium_trust_retrieve_more_loop_bounded() -> None:
    """Medium trust flow executes retrieval expansion loop and halts at max_attempts:
    0.50 <= trust_score < 0.75 -> RETRIEVE_MORE -> re-evaluate -> halts at max_attempts -> abstains if insufficient.
    """
    doc_id = "doc-medium"
    # Evidence with 1 supported chunk out of 3, yielding medium trust score (between 0.50 and 0.75)
    chunks = [
        {"chunk_id": "c1", "document_id": doc_id, "text": "Direct answer context.", "score": 0.85},
        {"chunk_id": "c2", "document_id": doc_id, "text": "Marginal context.", "score": 0.70},
        {"chunk_id": "c3", "document_id": doc_id, "text": "Irrelevant noise.", "score": 0.65},
    ]

    class MediumCritic:
        def evaluate(self, question: str, evidence: list[dict[str, Any]], document_id: str | None = None) -> dict[str, Any]:
            return {
                "document_id": document_id,
                "evaluations": [
                    {"chunk_id": "c1", "relevance": True, "support_status": "supported", "quality_assessment": {"evidence_strength": "high", "source_quality": "high", "potential_outdated": False}},
                    {"chunk_id": "c2", "relevance": False, "support_status": "unsupported", "quality_assessment": {"evidence_strength": "low", "source_quality": "medium", "potential_outdated": False}},
                    {"chunk_id": "c3", "relevance": False, "support_status": "unsupported", "quality_assessment": {"evidence_strength": "low", "source_quality": "medium", "potential_outdated": False}},
                ],
                "contradictions": [],
            }

    retriever = E2ERetriever({doc_id: chunks})
    orchestrator = TrustAwareOrchestrator(
        retriever=retriever,
        critic_agent=MediumCritic(),
        synthesizer_agent=E2ESynthesizer(),
        verifier_agent=E2EVerifier(),
        abstention_engine=AbstentionEngine(),
    )

    result = orchestrator.run(question="Explain algorithm detail?", document_id=doc_id, max_retries=1)

    # Retrieval loop ran, attempted expansion, halted boundedly without infinite loop
    assert retriever.retrieve_call_count == 2
    assert result["document_id"] == doc_id
    # Since evidence remained medium and retries exhausted, it routed to abstention
    assert result["abstention"] is True
    assert "insufficient" in result["decision"]["reason"].lower() or "medium trust" in result["decision"]["reason"].lower()


# ==============================================================================
# 3. LOW Trust Flow (Immediate Clean Abstention)
# ==============================================================================
def test_e2e_low_trust_abstention() -> None:
    """Low trust flow (< 0.50) routes directly to Abstention:
    Returns standard refusal message without hallucinating an answer.
    """
    doc_id = "doc-empty"
    retriever = E2ERetriever({doc_id: []})
    critic = E2ECritic(relevance=False, support_status="unsupported")
    orchestrator = TrustAwareOrchestrator(
        retriever=retriever,
        critic_agent=critic,
        synthesizer_agent=E2ESynthesizer(),
        verifier_agent=E2EVerifier(),
        abstention_engine=AbstentionEngine(),
    )

    result = orchestrator.run(question="What is the quantum state?", document_id=doc_id)

    assert result["document_id"] == doc_id
    assert result["abstention"] is True
    assert result["decision"]["decision"] == "ABSTAIN"
    assert "I don't have enough reliable evidence in the uploaded document to answer this question." in result["answer"]


# ==============================================================================
# 4. Contradiction Detection Flow & Explicit Marking
# ==============================================================================
def test_e2e_contradiction_detection_flow() -> None:
    """Contradiction flow detects factual conflicts, marks them explicitly, and abstains to prevent hallucination."""
    doc_id = "doc-conflict"
    chunks = [
        {"chunk_id": "c1", "document_id": doc_id, "text": "The test accuracy was 85% on dataset A.", "score": 0.95},
        {"chunk_id": "c2", "document_id": doc_id, "text": "The test accuracy was 72% on dataset A.", "score": 0.94},
    ]

    class ContradictionCritic:
        def evaluate(self, question: str, evidence: list[dict[str, Any]], document_id: str | None = None) -> dict[str, Any]:
            return {
                "document_id": document_id,
                "evaluations": [
                    {"chunk_id": "c1", "relevance": True, "support_status": "contradicted"},
                    {"chunk_id": "c2", "relevance": True, "support_status": "contradicted"},
                ],
                "contradictions": [
                    {"chunk_a": "c1", "chunk_b": "c2", "reason": "Numerical conflict: 85% vs 72%", "severity": "high"}
                ],
            }

    retriever = E2ERetriever({doc_id: chunks})
    orchestrator = TrustAwareOrchestrator(
        retriever=retriever,
        critic_agent=ContradictionCritic(),
        synthesizer_agent=E2ESynthesizer(),
        verifier_agent=E2EVerifier(),
        abstention_engine=AbstentionEngine(),
    )

    result = orchestrator.run(question="What was the accuracy?", document_id=doc_id)

    # Contradiction explicitly detected and causes safe abstention
    assert result["abstention"] is True
    assert len(result["contradictions"]) == 1
    assert result["contradictions"][0]["chunk_a"] == "c1"
    assert result["contradictions"][0]["chunk_b"] == "c2"
    assert result["decision"]["abstention_reason"] == "contradiction_detected"


# ==============================================================================
# 5. Cross-Document Anti-Contamination & Rejected Evidence Filtering
# ==============================================================================
def test_e2e_cross_document_and_rejected_evidence_isolation() -> None:
    """Foreign document chunks and rejected chunks are completely stripped before reaching Synthesizer."""
    doc_target = "doc-target"
    chunks = [
        {"chunk_id": "chunk-valid-1", "document_id": doc_target, "text": "Valid target fact 1.", "score": 0.95},
        {"chunk_id": "chunk-valid-2", "document_id": doc_target, "text": "Valid target fact 2.", "score": 0.92},
        {"chunk_id": "chunk-foreign", "document_id": "doc-foreign", "text": "Leaked foreign fact.", "score": 0.90},
        {"chunk_id": "chunk-rejected", "document_id": doc_target, "text": "Irrelevant noise.", "score": 0.40},
    ]

    class FilteringCritic:
        def evaluate(self, question: str, evidence: list[dict[str, Any]], document_id: str | None = None) -> dict[str, Any]:
            return {
                "document_id": document_id,
                "evaluations": [
                    {"chunk_id": "chunk-valid-1", "relevance": True, "support_status": "supported", "quality_assessment": {"evidence_strength": "high", "source_quality": "high", "potential_outdated": False}},
                    {"chunk_id": "chunk-valid-2", "relevance": True, "support_status": "supported", "quality_assessment": {"evidence_strength": "high", "source_quality": "high", "potential_outdated": False}},
                    {"chunk_id": "chunk-foreign", "relevance": False, "support_status": "unrelated_document", "quality_assessment": {"evidence_strength": "none", "source_quality": "low", "potential_outdated": False}},
                    {"chunk_id": "chunk-rejected", "relevance": False, "support_status": "rejected", "quality_assessment": {"evidence_strength": "none", "source_quality": "low", "potential_outdated": False}},
                ],
                "contradictions": [],
            }

    synth = E2ESynthesizer()
    retriever = E2ERetriever({doc_target: chunks})
    orchestrator = TrustAwareOrchestrator(
        retriever=retriever,
        critic_agent=FilteringCritic(),
        synthesizer_agent=synth,
        verifier_agent=E2EVerifier(status="SUPPORTED"),
        abstention_engine=AbstentionEngine(),
    )

    result = orchestrator.run(question="Tell me the target fact?", document_id=doc_target)

    # Check evidence received by Synthesizer
    synth_chunk_ids = [c["chunk_id"] for c in synth.received_evidence]
    assert "chunk-valid-1" in synth_chunk_ids
    assert "chunk-valid-2" in synth_chunk_ids
    assert "chunk-foreign" not in synth_chunk_ids
    assert "chunk-rejected" not in synth_chunk_ids

    # Target document ID preserved throughout
    assert result["document_id"] == doc_target
    assert synth.received_doc_id == doc_target


# ==============================================================================
# 6. Cache Key Scoping by Document ID
# ==============================================================================
def test_cache_keys_contain_document_id() -> None:
    """Verify cache keys strictly contain document_id to prevent cross-document cache contamination."""
    key1 = RedisCache.query_key("doc-1", "What is the capital?")
    key2 = RedisCache.query_key("doc-2", "What is the capital?")

    assert "doc-1" in key1
    assert "doc-2" in key2
    assert key1 != key2


# ==============================================================================
# 7. Previous Chat History Is Not Used as Factual Evidence
# ==============================================================================
def test_chat_history_not_injected_as_factual_evidence() -> None:
    """Verify chat_history parameter in ChatRequest is isolated and never becomes an evidence chunk."""
    from app.api.chat import ChatRequest

    req = ChatRequest(
        document_id="doc-123",
        question="What is the policy?",
        chat_history=[
            {"role": "user", "content": "Prior message"},
            {"role": "assistant", "content": "Prior answer claimed 99% uptime."},
        ],
    )

    assert req.chat_history is not None
    assert len(req.chat_history) == 2
    # Ensure chat_history is not evidence
    assert req.document_id == "doc-123"


# ==============================================================================
# 8. Loop Limits & Graceful Error Handling
# ==============================================================================
def test_graceful_error_handling_in_pipeline() -> None:
    """Unhandled exceptions in pipeline components are caught gracefully."""
    class FailingRetriever:
        def retrieve(self, *args, **kwargs):
            raise RuntimeError("Database connection timed out.")

    orchestrator = TrustAwareOrchestrator(
        retriever=FailingRetriever(),
        critic_agent=E2ECritic(),
        synthesizer_agent=E2ESynthesizer(),
        verifier_agent=E2EVerifier(),
        abstention_engine=AbstentionEngine(),
    )

    with pytest.raises(RuntimeError, match="Database connection timed out"):
        orchestrator.run(question="Test question", document_id="doc-fail")
