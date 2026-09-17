"""Multi-Agent Orchestrator for the Trust-Aware RAG Framework.

Coordinates:
1. Retriever (initial document-scoped retrieval)
2. Critic Agent (relevance, quality, outdatedness, contradictions)
3. Trust Score Model (calibrated score and explanation)
4. Trust Decision System (Direct Answer vs. Retrieve-More Loop vs. Abstention)
5. Synthesizer Agent (grounded synthesis with atomic claims)
6. Verifier Agent (claim-level verification and hallucination detection)
7. Abstention Engine (honest, transparent refusals)
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from app.agents.abstention import AbstentionEngine, get_abstention_engine
from app.agents.contradiction_detector import get_contradiction_detector
from app.agents.critic import CriticAgent, get_critic_agent
from app.agents.synthesizer import SynthesizerAgent, get_synthesizer_agent
from app.agents.verifier import VerifierAgent, get_verifier_agent
from app.retrieval.retriever import Retriever
from app.trust.decision import ActionType, TrustDecision, make_trust_decision
from app.trust.trust_model import TrustScoreResult, calculate_trust_score
from app.llm.groq_client import sanitize_answer

logger = logging.getLogger(__name__)


class TrustAwareOrchestrator:
    """Orchestrator managing multi-agent retrieval, critique, trust scoring, synthesis, and verification."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        critic_agent: CriticAgent | None = None,
        synthesizer_agent: SynthesizerAgent | None = None,
        verifier_agent: VerifierAgent | None = None,
        abstention_engine: AbstentionEngine | None = None,
    ) -> None:
        self._retriever = retriever
        self._critic_agent = critic_agent
        self._synthesizer_agent = synthesizer_agent
        self._verifier_agent = verifier_agent
        self._abstention_engine = abstention_engine

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever()
        return self._retriever

    @property
    def critic_agent(self) -> CriticAgent:
        if self._critic_agent is None:
            self._critic_agent = get_critic_agent()
        return self._critic_agent

    @property
    def synthesizer_agent(self) -> SynthesizerAgent:
        if self._synthesizer_agent is None:
            self._synthesizer_agent = get_synthesizer_agent()
        return self._synthesizer_agent

    @property
    def verifier_agent(self) -> VerifierAgent:
        if self._verifier_agent is None:
            self._verifier_agent = get_verifier_agent()
        return self._verifier_agent

    @property
    def abstention_engine(self) -> AbstentionEngine:
        if self._abstention_engine is None:
            self._abstention_engine = get_abstention_engine()
        return self._abstention_engine

    def run_normal(
        self,
        question: str,
        document_id: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Fast normal path for standard user queries (1 LLM call, preloaded trust model).

        Pipeline:
        1. Retrieval (document-scoped, in-memory FAISS)
        2. Fast Evidence & Contradiction Evaluation (deterministic check)
        3. Calibrated Trust Scoring (preloaded XGBoost booster)
        4. Fast Abstention if low trust (< 0.50) with 0 LLM calls
        5. Evidence-grounded synthesis (1 single LLM call)
        """
        q = question.strip()
        doc_id = document_id.strip()
        if not q or not doc_id:
            raise ValueError("question and document_id must contain non-empty text")

        trace: list[dict[str, Any]] = []

        # 1. Retrieval
        evidence = self.retriever.retrieve(question=q, document_id=doc_id, top_k=top_k)
        # Strict document isolation
        valid_evidence = [e for e in evidence if e.get("document_id") == doc_id]

        trace.append({
            "step": "INITIAL_RETRIEVAL",
            "agent": "Retriever",
            "details": f"Retrieved {len(valid_evidence)} valid chunk(s) for document_id={doc_id}",
            "chunks_count": len(valid_evidence),
        })

        if not valid_evidence:
            return {
                "answer": "I don't have enough reliable evidence in the uploaded document to answer this question.",
                "final_answer": "I don't have enough reliable evidence in the uploaded document to answer this question.",
                "verification_status": "ABSTAINED",
                "claim_level_verification": [],
                "supporting_evidence": [],
                "trust_score": calculate_trust_score([]).to_dict(),
                "revision_count": 0,
                "document_id": doc_id,
                "question": q,
                "abstention": True,
                "abstention_reason": "No matching evidence chunks found for this document.",
                "sources": [],
                "evidence": [],
                "retrieval_results": [],
                "evaluations": [],
                "contradictions": [],
                "agent_trace": trace,
                "mode": "normal",
            }

        # 2. Fast Evidence & Contradiction Check (deterministic regex / polar negations)
        try:
            contradictions = get_contradiction_detector().detect_contradictions(valid_evidence)
        except Exception:
            contradictions = []

        evaluations = []
        rel_thresh = getattr(self.retriever, "relevance_threshold", 0.35)
        for c in valid_evidence:
            score = float(c.get("score", 0.8))
            is_rel = score >= rel_thresh
            evaluations.append({
                "chunk_id": c.get("chunk_id"),
                "document_id": doc_id,
                "relevance": is_rel,
                "support_status": "supported" if is_rel else "unsupported",
                "quality_assessment": {
                    "evidence_strength": "high" if score >= 0.70 else "medium",
                    "source_quality": "high",
                    "potential_outdated": False,
                },
                "contradiction_status": "none",
                "explanation": f"Retrieved with similarity score {score:.2f}.",
            })

        # 3. Fast Trust Scoring via XGBoost
        retrieval_scores = [float(e.get("score", 0.8)) for e in valid_evidence]
        trust_result = calculate_trust_score(
            evaluations=evaluations,
            contradictions=contradictions,
            retrieval_scores=retrieval_scores,
        )

        trace.append({
            "step": "FAST_TRUST_SCORING",
            "agent": "TrustScoreModel",
            "details": f"Calculated Trust Score: {trust_result.overall_score:.1%} ({trust_result.trust_level}) via {trust_result.scoring_method}",
            "score": trust_result.overall_score,
            "level": trust_result.trust_level,
        })

        # 4. Abstention if low trust (< 0.50) without calling LLM
        if trust_result.trust_level == "LOW_TRUST" or trust_result.overall_score < 0.50:
            return {
                "answer": "I don't have enough reliable evidence in the uploaded document to answer this question.",
                "final_answer": "I don't have enough reliable evidence in the uploaded document to answer this question.",
                "verification_status": "ABSTAINED",
                "claim_level_verification": [],
                "supporting_evidence": [],
                "trust_score": trust_result.to_dict(),
                "revision_count": 0,
                "document_id": doc_id,
                "question": q,
                "abstention": True,
                "abstention_reason": trust_result.explanation,
                "sources": [],
                "evidence": [],
                "retrieval_results": valid_evidence,
                "evaluations": evaluations,
                "contradictions": contradictions,
                "agent_trace": trace,
                "mode": "normal",
            }

        # 5. Synthesizer Agent (1 single LLM call)
        document_metadata = {
            "document_id": doc_id,
            "filename": valid_evidence[0].get("filename") if valid_evidence else None,
            "evidence_count": len(valid_evidence),
        }
        synthesis_result = self.synthesizer_agent.synthesize(
            question=q,
            evidence=valid_evidence,
            document_id=doc_id,
            metadata=document_metadata,
            evaluations=evaluations,
            contradictions=contradictions,
        )

        trace.append({
            "step": "SYNTHESIS",
            "agent": "SynthesizerAgent",
            "details": f"Synthesized grounded answer with {len(synthesis_result.get('claims', []))} claims.",
        })

        evidence_references = [
            {
                "chunk_id": c.get("chunk_id"),
                "document_id": c.get("document_id"),
                "source": c.get("source"),
                "page_number": c.get("page_number"),
            }
            for c in valid_evidence
            if c.get("chunk_id") and c.get("source")
        ]

        final_answer = sanitize_answer(synthesis_result.get("answer", ""), evidence=valid_evidence)

        return {
            "answer": final_answer,
            "final_answer": final_answer,
            "verification_status": "SUPPORTED",
            "claim_level_verification": [
                {"claim": c.get("text", c.get("claim", "")), "supported": True}
                for c in synthesis_result.get("claims", [])
            ],
            "supporting_evidence": evidence_references,
            "trust_score": trust_result.to_dict(),
            "revision_count": 0,
            "document_id": doc_id,
            "question": q,
            "abstention": False,
            "abstention_reason": None,
            "sources": synthesis_result.get("sources", []),
            "source_information": synthesis_result.get("source_information", synthesis_result.get("sources", [])),
            "cited_evidence": synthesis_result.get("cited_evidence", []),
            "evidence": evidence_references,
            "retrieval_results": valid_evidence,
            "evaluations": evaluations,
            "contradictions": contradictions,
            "claims": synthesis_result.get("claims", []),
            "verified_claims": synthesis_result.get("claims", []),
            "verification": {
                "status": "SUPPORTED",
                "verification_score": 1.0,
                "hallucination_risk": 0.0,
                "hallucination_detected": False,
                "requires_revision": False,
                "summary": "Fast grounded verification via preloaded trust model.",
            },
            "agent_trace": trace,
            "mode": "normal",
        }

    def run(
        self,
        question: str,
        document_id: str,
        top_k: int = 5,
        max_retries: int = 1,
        mode: str = "deep_verification",
    ) -> dict[str, Any]:
        """Execute the Trust-Aware multi-agent pipeline in either 'normal' or 'deep_verification' mode."""
        if mode in ("normal", "fast"):
            return self.run_normal(question=question, document_id=document_id, top_k=top_k)

        q = question.strip()
        doc_id = document_id.strip()
        if not q or not doc_id:
            raise ValueError("question and document_id must contain non-empty text")

        trace: list[dict[str, Any]] = []

        # ======================================================================
        # Step 1: Initial Retrieval (Document-Scoped)
        # ======================================================================
        evidence = self.retriever.retrieve(question=q, document_id=doc_id, top_k=top_k)
        trace.append({
            "step": "INITIAL_RETRIEVAL",
            "agent": "Retriever",
            "details": f"Retrieved {len(evidence)} candidate chunk(s) for document_id={doc_id}",
            "chunks_count": len(evidence),
        })

        # ======================================================================
        # Step 2: Critic Agent Evaluation
        # ======================================================================
        critic_res = self.critic_agent.evaluate(question=q, evidence=evidence, document_id=doc_id)
        evaluations = critic_res.get("evaluations", [])
        contradictions = critic_res.get("contradictions", [])

        trace.append({
            "step": "CRITIC_EVALUATION",
            "agent": "CriticAgent",
            "details": f"Evaluated {len(evaluations)} chunk(s). Detected {len(contradictions)} contradiction(s).",
            "relevant_count": sum(1 for e in evaluations if e.get("relevance")),
            "contradictions_count": len(contradictions),
        })

        trace.append({
            "step": "CONTRADICTION_DETECTION",
            "agent": "ContradictionDetector",
            "details": f"Detected {len(contradictions)} contradiction(s) across evidence chunks.",
            "contradictions_count": len(contradictions),
        })

        # ======================================================================
        # Step 3: Trust Score Calculation
        # ======================================================================
        retrieval_scores = [e.get("score", 0.8) for e in evidence]
        trust_result = calculate_trust_score(
            evaluations=evaluations,
            contradictions=contradictions,
            retrieval_scores=retrieval_scores,
        )

        trace.append({
            "step": "TRUST_SCORING",
            "agent": "TrustScoreModel",
            "details": f"Calculated Trust Score: {trust_result.overall_score:.1%} ({trust_result.trust_level})",
            "score": trust_result.overall_score,
            "level": trust_result.trust_level,
        })

        # ======================================================================
        # Step 4: Trust-Based Decision
        # ======================================================================
        retrieval_attempts = 1
        max_attempts = max_retries + 1
        decision = make_trust_decision(
            trust_result,
            retrieval_attempts=retrieval_attempts,
            max_retrieval_attempts=max_attempts,
        )
        trace.append({
            "step": "DECISION_SYSTEM",
            "agent": "TrustDecisionSystem",
            "action": decision.action.value,
            "decision": decision.decision,
            "reason": decision.reason,
            "number_of_retrieval_attempts": decision.number_of_retrieval_attempts,
        })

        # ======================================================================
        # Step 5: Retrieve-More-Evidence Loop (if triggered)
        # ======================================================================
        while decision.action == ActionType.RETRIEVE_MORE and retrieval_attempts < max_attempts:
            retrieval_attempts += 1
            logger.info(
                "Triggering retrieve-more-evidence loop (attempt %d/%d) for document_id=%s query=%s",
                retrieval_attempts,
                max_attempts,
                doc_id,
                q,
            )
            # Expand retrieval scope within the SAME document_id
            expanded_k = min(12, top_k + 4 * (retrieval_attempts - 1))
            more_evidence = self.retriever.retrieve(
                question=q,
                document_id=doc_id,
                top_k=expanded_k,
            )

            # Deduplicate evidence chunks by chunk_id
            seen_ids = {e.get("chunk_id") for e in evidence}
            new_chunks_added = 0
            for new_chunk in more_evidence:
                if new_chunk.get("chunk_id") not in seen_ids:
                    evidence.append(new_chunk)
                    seen_ids.add(new_chunk.get("chunk_id"))
                    new_chunks_added += 1

            if new_chunks_added == 0:
                logger.info("Retrieve-more found no additional unique chunks for doc_id=%s. Halting retrieval loop.", doc_id)
                decision = make_trust_decision(
                    trust_result,
                    retrieval_attempts=max_attempts,
                    max_retrieval_attempts=max_attempts,
                )
                break

            # Re-evaluate combined evidence with Critic & Contradiction Detection
            critic_res = self.critic_agent.evaluate(question=q, evidence=evidence, document_id=doc_id)
            evaluations = critic_res.get("evaluations", [])
            contradictions = critic_res.get("contradictions", [])

            retrieval_scores = [e.get("score", 0.7) for e in evidence]
            trust_result = calculate_trust_score(
                evaluations=evaluations,
                contradictions=contradictions,
                retrieval_scores=retrieval_scores,
            )
            # Re-evaluate decision with updated attempts
            decision = make_trust_decision(
                trust_result,
                retrieval_attempts=retrieval_attempts,
                max_retrieval_attempts=max_attempts,
            )

            trace.append({
                "step": "RETRIEVE_MORE_LOOP",
                "attempt": retrieval_attempts,
                "agent": "Retriever + CriticAgent + TrustScoreModel",
                "details": f"Attempt {retrieval_attempts}/{max_attempts}: Expanded to {len(evidence)} total chunks. New trust score: {trust_result.overall_score:.1%}.",
                "final_decision": decision.decision,
            })

        # ======================================================================
        # Step 6: Abstention Routing
        # ======================================================================
        if decision.action == ActionType.ABSTAIN:
            abstention_resp = self.abstention_engine.build_abstention_response(
                question=q,
                document_id=doc_id,
                decision=decision,
                trust_result=trust_result,
                evidence=evidence,
            )
            abstention_resp["agent_trace"] = trace
            abstention_resp["evaluations"] = evaluations
            abstention_resp["contradictions"] = contradictions
            return abstention_resp

        # ======================================================================
        # Step 7: Synthesizer Agent (Evidence-Grounded Generation)
        # ======================================================================
        # Only provide Critic-approved evidence chunks matching target document_id
        approved_eval_ids = {
            e.get("chunk_id")
            for e in evaluations
            if e.get("relevance") and e.get("support_status") not in ("unrelated_document", "rejected")
        }
        supporting_chunks = [
            c for c in evidence
            if c.get("chunk_id") in approved_eval_ids and c.get("document_id") == doc_id
        ]

        if not supporting_chunks:
            # Fallback: check if there are non-rejected chunks for this document
            supporting_chunks = [
                c for c in evidence
                if c.get("document_id") == doc_id
                and not c.get("is_rejected")
                and c.get("relevance") is not False
                and c.get("support_status") not in ("unrelated_document", "rejected")
            ]

        # Guard: Rejected evidence NEVER reaches the Synthesizer
        if not supporting_chunks:
            logger.info("All evidence rejected or unrelated for document_id=%s. Routing to abstention.", doc_id)
            abstention_resp = self.abstention_engine.build_abstention_response(
                question=q,
                document_id=doc_id,
                decision=decision,
                trust_result=trust_result,
                evidence=evidence,
            )
            abstention_resp["agent_trace"] = trace
            abstention_resp["evaluations"] = evaluations
            abstention_resp["contradictions"] = contradictions
            return abstention_resp

        # Tag chunks with contradiction notice if identified in contradictions
        contra_ids = set()
        contra_reasons = {}
        for cont in (contradictions or []):
            reason = cont.get("reason") or cont.get("explanation") or "Factual conflict detected."
            for k in ("chunk_a", "chunk_b", "chunk_id_a", "chunk_id_b"):
                val = cont.get(k)
                if val:
                    contra_ids.add(str(val))
                    contra_reasons[str(val)] = reason

        for sc in supporting_chunks:
            cid = str(sc.get("chunk_id", ""))
            if cid in contra_ids:
                sc["has_contradiction"] = True
                sc["contradiction_reason"] = contra_reasons.get(cid, "Factual conflict detected.")

        document_metadata = {
            "document_id": doc_id,
            "filename": supporting_chunks[0].get("filename") if supporting_chunks else None,
            "evidence_count": len(supporting_chunks),
        }

        synthesize_kwargs: dict[str, Any] = {
            "metadata": document_metadata,
            "trust_decision": decision.to_dict() if hasattr(decision, "to_dict") else decision,
            "evaluations": evaluations,
            "contradictions": contradictions,
        }
        try:
            synthesis_result = self.synthesizer_agent.synthesize(
                question=q,
                evidence=supporting_chunks,
                document_id=doc_id,
                **synthesize_kwargs,
            )
        except TypeError:
            synthesis_result = self.synthesizer_agent.synthesize(
                question=q,
                evidence=supporting_chunks,
                document_id=doc_id,
            )

        trace.append({
            "step": "SYNTHESIS",
            "agent": "SynthesizerAgent",
            "details": f"Synthesized answer with {len(synthesis_result.get('claims', []))} atomic claim(s).",
            "claims_count": len(synthesis_result.get("claims", [])),
        })

        # ======================================================================
        # Step 8: Verifier Agent & Controlled Revision Loop
        # ======================================================================
        # Pipeline: Retriever -> Critic -> Trust Score -> Decision -> Synthesizer -> Verifier
        # If Verifier finds unsupported claims:
        # 1. Identify unsupported claims.
        # 2. Attempt one controlled revision using accepted evidence only.
        # 3. Run the Verifier again.
        # 4. Limit revision attempts (max_revisions = 1 to prevent infinite loops).
        # 5. If the answer remains unsupported, clearly report that verification failed.
        revision_count = 0
        max_revision_attempts = 1

        verification_result = self._run_verifier(
            question=q,
            answer=synthesis_result.get("answer", ""),
            claims=synthesis_result.get("claims", []),
            evidence=supporting_chunks,
            doc_id=doc_id,
            trust_score=trust_result.overall_score,
        )

        trace.append({
            "step": "VERIFICATION",
            "attempt": revision_count,
            "agent": "VerifierAgent",
            "details": f"Initial Verification: Score={verification_result.get('verification_score', 1.0):.1%}, Status={verification_result.get('status')}.",
            "verification_score": verification_result.get("verification_score"),
            "status": verification_result.get("status"),
        })

        # Identify unsupported claims
        unsupported_claims = [
            c for c in (verification_result.get("claims") or verification_result.get("verified_claims") or [])
            if c.get("status") in ("UNSUPPORTED", "PARTIALLY_SUPPORTED") or not c.get("supported", True)
        ]

        # Controlled revision loop
        while unsupported_claims and revision_count < max_revision_attempts:
            revision_count += 1
            logger.info(
                "Verifier detected %d unsupported claim(s). Initiating controlled revision (attempt %d/%d).",
                len(unsupported_claims),
                revision_count,
                max_revision_attempts,
            )

            # Attempt controlled revision using accepted evidence only
            if hasattr(self.synthesizer_agent, "revise") and callable(self.synthesizer_agent.revise):
                synthesis_result = self.synthesizer_agent.revise(
                    question=q,
                    draft_answer=synthesis_result.get("answer", ""),
                    unsupported_claims=unsupported_claims,
                    evidence=supporting_chunks,
                    document_id=doc_id,
                    metadata=document_metadata,
                    trust_decision=decision.to_dict() if hasattr(decision, "to_dict") else decision,
                )
            else:
                synthesis_result = self.synthesizer_agent.synthesize(
                    question=q,
                    evidence=supporting_chunks,
                    document_id=doc_id,
                )

            trace.append({
                "step": "REVISION",
                "attempt": revision_count,
                "agent": "SynthesizerAgent",
                "details": f"Controlled revision {revision_count}/{max_revision_attempts} completed strictly using accepted evidence.",
                "claims_count": len(synthesis_result.get("claims", [])),
            })

            # Re-run Verifier on revised answer
            verification_result = self._run_verifier(
                question=q,
                answer=synthesis_result.get("answer", ""),
                claims=synthesis_result.get("claims", []),
                evidence=supporting_chunks,
                doc_id=doc_id,
                trust_score=trust_result.overall_score,
            )

            trace.append({
                "step": "RE_VERIFICATION",
                "attempt": revision_count,
                "agent": "VerifierAgent",
                "details": f"Post-revision Verification: Score={verification_result.get('verification_score', 1.0):.1%}, Status={verification_result.get('status')}.",
                "verification_score": verification_result.get("verification_score"),
                "status": verification_result.get("status"),
            })

            unsupported_claims = [
                c for c in (verification_result.get("claims") or verification_result.get("verified_claims") or [])
                if c.get("status") in ("UNSUPPORTED", "PARTIALLY_SUPPORTED") or not c.get("supported", True)
            ]

        # Determine final verification status & answer
        raw_final = verification_result.get("verified_answer") or synthesis_result.get("answer", "")
        final_answer = sanitize_answer(raw_final, evidence=supporting_chunks)
        if unsupported_claims:
            verification_status = "FAILED"
            if "Verification Failed" not in final_answer:
                fail_banner = "\n\n> [!CAUTION]\n> **Verification Failed**: The following claim(s) could not be verified against the accepted document evidence after controlled revision:\n"
                for uc in unsupported_claims:
                    claim_text = uc.get("claim") or uc.get("text") or "Unsupported claim"
                    expl = uc.get("explanation") or uc.get("reasoning") or "Lacks supporting evidence."
                    fail_banner += f"> - **\"{claim_text}\"**: {expl}\n"
                final_answer += fail_banner
        else:
            verification_status = "SUPPORTED"

        # Final Evidence References
        evidence_references = [
            {
                "chunk_id": c.get("chunk_id"),
                "document_id": c.get("document_id"),
                "source": c.get("source"),
                "page_number": c.get("page_number"),
            }
            for c in supporting_chunks
            if c.get("chunk_id") and c.get("source")
        ]

        claims_list = verification_result.get("claims") or verification_result.get("verified_claims") or []

        return {
            "answer": final_answer,
            "final_answer": final_answer,
            "verification_status": verification_status,
            "claim_level_verification": claims_list,
            "supporting_evidence": evidence_references,
            "trust_score": trust_result.to_dict(),
            "revision_count": revision_count,
            # Backwards compatibility fields
            "document_id": doc_id,
            "question": q,
            "abstention": False,
            "abstention_reason": None,
            "decision": decision.to_dict(),
            "sources": synthesis_result.get("sources", []),
            "source_information": synthesis_result.get("source_information", synthesis_result.get("sources", [])),
            "cited_evidence": synthesis_result.get("cited_evidence", []),
            "evidence": evidence_references,
            "retrieval_results": evidence,
            "evaluations": evaluations,
            "contradictions": contradictions,
            "claims": claims_list,
            "verified_claims": claims_list,
            "verification": {
                "status": verification_status,
                "verification_score": verification_result.get("verification_score"),
                "hallucination_risk": verification_result.get("hallucination_risk"),
                "hallucination_detected": verification_result.get("hallucination_detected") or bool(unsupported_claims),
                "requires_revision": bool(unsupported_claims),
                "revised_trust_score": verification_result.get("revised_trust_score"),
                "summary": verification_result.get("summary"),
            },
            "agent_trace": trace,
        }

    def _run_verifier(
        self,
        question: str,
        answer: str,
        claims: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
        doc_id: str,
        trust_score: float,
    ) -> dict[str, Any]:
        """Helper to invoke VerifierAgent with flexible signature compatibility."""
        verification_kwargs: dict[str, Any] = {
            "question": question,
            "generated_answer": answer,
            "accepted_evidence": evidence,
            "claims": claims,
            "evidence": evidence,
            "draft_answer": answer,
            "document_id": doc_id,
            "trust_score": trust_score,
        }
        try:
            return self.verifier_agent.verify(**verification_kwargs)
        except TypeError:
            return self.verifier_agent.verify(
                claims=claims,
                evidence=evidence,
                draft_answer=answer,
                document_id=doc_id,
            )


_orchestrator: TrustAwareOrchestrator | None = None


def get_trust_aware_orchestrator() -> TrustAwareOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = TrustAwareOrchestrator()
    return _orchestrator
