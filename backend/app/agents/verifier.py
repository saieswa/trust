"""Verifier Agent for Claim-Level Verification in Trust-Aware RAG.

Performs independent verification of each meaningful claim in the generated answer
against the accepted evidence chunks to detect, measure, and eliminate hallucinations.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from app.llm.groq_client import GroqClientError, GroqLLMClient, get_groq_client

logger = logging.getLogger(__name__)

VERIFIER_SYSTEM_PROMPT = """You are an expert Verifier Agent in a Trust-Aware RAG system.
Your mission is to rigorously verify each meaningful claim in the generated answer against ONLY the accepted evidence chunks provided.

STRICT VERIFICATION STANDARDS:
1. "SUPPORTED": The claim is directly, fully, and factually substantiated by the accepted evidence chunks.
2. "PARTIALLY_SUPPORTED": The claim is partially substantiated by the accepted evidence, but contains unverified nuances, extrapolations, or ungrounded details.
3. "UNSUPPORTED": The claim is not substantiated, lacks evidence in the accepted chunks, hallucinates unstated facts, contradicts the evidence, or relies on external knowledge not present in the current document's accepted evidence.

STRICT RULES:
- Do NOT use outside, general, or pre-trained knowledge as evidence.
- A claim must be marked "UNSUPPORTED" if the provided accepted evidence chunks do not directly state it, even if the claim is true in the real world.
- Evidence from other documents CANNOT support claims for the current document.
- For each claim, return whether it is supported (true only if SUPPORTED), its status, supporting chunk IDs, and a concise explanation.

You must return ONLY valid JSON matching this schema:
{
  "claim_evaluations": [
    {
      "claim_id": "<claim_id>",
      "claim": "<claim text>",
      "status": "SUPPORTED" | "PARTIALLY_SUPPORTED" | "UNSUPPORTED",
      "supported": true | false,
      "supporting_chunk_ids": ["<chunk_id>"],
      "explanation": "<concise justification based strictly on the accepted evidence>"
    }
  ]
}
"""


@dataclass(frozen=True)
class ClaimVerificationResult:
    claim_id: str
    claim: str
    supported: bool
    status: str  # SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED
    supporting_chunk_ids: list[str]
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["text"] = self.claim
        d["reasoning"] = self.explanation
        return d


class VerifierAgent:
    """Agent responsible for claim-level verification against accepted evidence."""

    def __init__(
        self,
        llm_client: GroqLLMClient | Any | None = None,
        model: str | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._model = model

    @property
    def llm_client(self) -> Any:
        if self._llm_client is None:
            self._llm_client = get_groq_client()
        return self._llm_client

    @property
    def model(self) -> str:
        if self._model:
            return self._model
        return getattr(self.llm_client, "model", "openai/gpt-oss-20b")

    def verify(
        self,
        question: str | None = None,
        generated_answer: str | None = None,
        accepted_evidence: Sequence[Mapping[str, Any]] | None = None,
        *,
        claims: Sequence[Mapping[str, Any]] | None = None,
        evidence: Sequence[Mapping[str, Any]] | None = None,
        draft_answer: str | None = None,
        document_id: str | None = None,
        trust_score: float | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Verify each meaningful claim against accepted evidence."""
        # Resolve inputs supporting both new API and legacy signatures
        answer_text = (generated_answer if generated_answer is not None else (draft_answer or "")).strip()
        evidence_items = accepted_evidence if accepted_evidence is not None else (evidence or [])
        user_question = (question or "").strip()

        # Step 1: Normalize and isolate accepted evidence for document_id
        target_doc_id = document_id
        if not target_doc_id:
            # Infer target document_id from evidence if present
            for item in evidence_items:
                c = dict(item.__dict__) if hasattr(item, "__dict__") else dict(item)
                if c.get("document_id"):
                    target_doc_id = str(c.get("document_id"))
                    break

        valid_chunk_map: dict[str, dict[str, Any]] = {}
        cross_doc_chunk_ids: set[str] = set()

        for item in evidence_items:
            c = dict(item.__dict__) if hasattr(item, "__dict__") else dict(item)
            cid = str(c.get("chunk_id", "")).strip()
            c_doc = str(c.get("document_id", "")).strip()
            if not cid:
                continue

            # Check cross-document leakage
            if target_doc_id and c_doc and c_doc != target_doc_id:
                cross_doc_chunk_ids.add(cid)
                logger.warning("Verifier filtered chunk %s: document mismatch (%s != %s)", cid, c_doc, target_doc_id)
                continue
            if c.get("support_status") == "unrelated_document":
                cross_doc_chunk_ids.add(cid)
                continue

            valid_chunk_map[cid] = c

        # Step 2: Extract meaningful claims if claims not provided
        if claims is not None:
            candidate_claims = list(claims)
        elif answer_text:
            candidate_claims = self._extract_meaningful_claims(answer_text)
        else:
            candidate_claims = []

        if not candidate_claims:
            return {
                "status": "SUPPORTED",
                "verification_score": 1.0,
                "requires_revision": False,
                "hallucination_risk": "LOW",
                "hallucination_detected": False,
                "claims": [],
                "verified_claims": [],
                "verified_answer": answer_text,
                "revised_trust_score": self._resolve_trust_score(trust_score),
                "summary": {
                    "total_claims": 0,
                    "supported_claims": 0,
                    "partially_supported_claims": 0,
                    "unsupported_claims": 0,
                },
            }

        # Step 3: Run claim evaluations with deterministic checks & LLM
        evals = self._evaluate_claims(
            claims=candidate_claims,
            chunk_map=valid_chunk_map,
            cross_doc_chunk_ids=cross_doc_chunk_ids,
            question=user_question,
            target_document_id=target_doc_id,
        )

        # Step 4: Aggregate results
        verified_claims: list[dict[str, Any]] = []
        for ev in evals:
            cid = ev.get("claim_id", "")
            claim_text = ev.get("claim") or ev.get("text", "")
            status = str(ev.get("status", "UNSUPPORTED")).upper()
            if status not in ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"):
                status = "UNSUPPORTED"
            supported = bool(ev.get("supported", status == "SUPPORTED"))
            supporting_ids = [str(x) for x in ev.get("supporting_chunk_ids", [])]
            explanation = str(ev.get("explanation") or ev.get("reasoning", ""))

            verified_claims.append({
                "claim_id": cid,
                "claim": claim_text,
                "text": claim_text,  # Backward compatibility
                "supported": supported,
                "status": status,
                "supporting_chunk_ids": supporting_ids,
                "explanation": explanation,
                "reasoning": explanation,  # Backward compatibility
                "confidence": float(ev.get("confidence", 0.9 if supported else 0.2)),
            })

        total = len(verified_claims)
        supported_count = sum(1 for c in verified_claims if c["status"] == "SUPPORTED")
        partially_count = sum(1 for c in verified_claims if c["status"] == "PARTIALLY_SUPPORTED")
        unsupported_count = sum(1 for c in verified_claims if c["status"] == "UNSUPPORTED")

        # Score calculation: SUPPORTED=1.0, PARTIALLY_SUPPORTED=0.5, UNSUPPORTED=0.0
        verification_score = round((supported_count + 0.5 * partially_count) / total, 3) if total > 0 else 1.0

        # Status categorization
        if unsupported_count > 0:
            overall_status = "UNSUPPORTED"
            hallucination_detected = True
            hallucination_risk = "HIGH"
            requires_revision = True
        elif partially_count > 0:
            overall_status = "PARTIALLY_SUPPORTED"
            hallucination_detected = True
            hallucination_risk = "MEDIUM"
            requires_revision = True
        else:
            overall_status = "SUPPORTED"
            hallucination_detected = False
            hallucination_risk = "LOW"
            requires_revision = False

        # Lower trust score if unsupported or partially supported
        original_trust = self._resolve_trust_score(trust_score)
        if requires_revision:
            if unsupported_count > 0:
                revised_trust = min(original_trust * verification_score, 0.40)
            else:
                revised_trust = min(original_trust * verification_score, 0.65)
        else:
            revised_trust = original_trust
        revised_trust = round(revised_trust, 3)

        # Clearly mark unsupported / partially supported claims in verified answer
        annotated_answer = answer_text
        if requires_revision:
            flagged = [c for c in verified_claims if c["status"] != "SUPPORTED"]
            banner = "\n\n> [!WARNING]\n> **Verifier Note (Revision Required)**: The following claim(s) are not fully substantiated by the accepted document evidence:\n"
            for fc in flagged:
                banner += f"> - **\"{fc['claim']}\"** [{fc['status']}]: {fc['explanation']}\n"
            annotated_answer = answer_text + banner

        return {
            "status": overall_status,
            "verification_score": verification_score,
            "requires_revision": requires_revision,
            "revised_trust_score": revised_trust,
            "hallucination_detected": hallucination_detected,
            "hallucination_risk": hallucination_risk,
            "claims": verified_claims,
            "verified_claims": verified_claims,  # Backward compatibility
            "verified_answer": annotated_answer,
            "summary": {
                "total_claims": total,
                "supported_claims": supported_count,
                "partially_supported_claims": partially_count,
                "unsupported_claims": unsupported_count,
            },
        }

    def _extract_meaningful_claims(self, text: str) -> list[dict[str, Any]]:
        """Split generated answer into atomic claim statements."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        raw_sentences: list[str] = []
        for line in lines:
            if line.startswith("#"):
                continue
            clean_line = re.sub(r"^[-*•\d\.]+\s*", "", line).strip()
            if not clean_line:
                continue
            sents = re.split(r"(?<=[.!?])\s+", clean_line)
            for s in sents:
                sc = s.strip()
                if len(sc) > 10 and not sc.startswith("#") and not sc.startswith(">"):
                    raw_sentences.append(sc)

        if not raw_sentences and text.strip():
            raw_sentences = [text.strip()]

        return [
            {"claim_id": f"claim-{i+1}", "claim": s, "text": s, "cited_chunk_ids": []}
            for i, s in enumerate(raw_sentences)
        ]

    def _evaluate_claims(
        self,
        claims: Sequence[Mapping[str, Any]],
        chunk_map: dict[str, dict[str, Any]],
        cross_doc_chunk_ids: set[str],
        question: str,
        target_document_id: str | None,
    ) -> list[dict[str, Any]]:
        """Evaluate claims using deterministic checks where applicable and LLM for semantic verification."""
        evaluations: list[dict[str, Any]] = []
        claims_needing_llm: list[dict[str, Any]] = []

        # Deterministic checks first
        for item in claims:
            c = dict(item.__dict__) if hasattr(item, "__dict__") else dict(item)
            cid = str(c.get("claim_id", ""))
            claim_text = str(c.get("claim") or c.get("text", "")).strip()
            cited_ids = [str(x) for x in c.get("cited_chunk_ids", [])]

            # Deterministic Check 1: Empty accepted evidence
            if not chunk_map:
                evaluations.append({
                    "claim_id": cid,
                    "claim": claim_text,
                    "status": "UNSUPPORTED",
                    "supported": False,
                    "supporting_chunk_ids": [],
                    "explanation": "No accepted evidence chunks available in the current document to support this claim.",
                })
                continue

            # Deterministic Check 2: Claim cites ONLY chunks from another document
            if cited_ids and all(x in cross_doc_chunk_ids for x in cited_ids):
                evaluations.append({
                    "claim_id": cid,
                    "claim": claim_text,
                    "status": "UNSUPPORTED",
                    "supported": False,
                    "supporting_chunk_ids": [],
                    "explanation": "Claim cites evidence belonging to another document. Cross-document evidence cannot support claims in the current document.",
                })
                continue

            # Deterministic Check 3: Check obvious numerical / negation contradiction against chunks
            numerical_conflict, reason = self._detect_deterministic_conflict(claim_text, chunk_map)
            if numerical_conflict:
                evaluations.append({
                    "claim_id": cid,
                    "claim": claim_text,
                    "status": "UNSUPPORTED",
                    "supported": False,
                    "supporting_chunk_ids": [],
                    "explanation": reason,
                })
                continue

            # Otherwise queue for LLM evaluation
            claims_needing_llm.append({
                "claim_id": cid,
                "claim": claim_text,
                "text": claim_text,
                "cited_chunk_ids": cited_ids,
            })

        if claims_needing_llm:
            llm_results = self._call_llm_for_verification(claims_needing_llm, chunk_map, question)
            llm_map_by_id = {e.get("claim_id"): e for e in llm_results if isinstance(e, dict) and e.get("claim_id")}
            llm_map_by_text = {e.get("claim", "").strip().lower(): e for e in llm_results if isinstance(e, dict) and e.get("claim")}

            for idx, c in enumerate(claims_needing_llm):
                cid = c["claim_id"]
                claim_text = c["claim"]
                cited_ids = c["cited_chunk_ids"]

                ev = None
                if cid in llm_map_by_id:
                    ev = llm_map_by_id[cid]
                elif claim_text.strip().lower() in llm_map_by_text:
                    ev = llm_map_by_text[claim_text.strip().lower()]
                elif idx < len(llm_results) and isinstance(llm_results[idx], dict):
                    ev = llm_results[idx]

                if ev:
                    status = str(ev.get("status", "UNSUPPORTED")).upper()
                    if status not in ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"):
                        status = "UNSUPPORTED"
                    supported = bool(ev.get("supported", status == "SUPPORTED"))
                    sup_ids = [str(x) for x in ev.get("supporting_chunk_ids", cited_ids) if str(x) in chunk_map]
                    evaluations.append({
                        "claim_id": cid,
                        "claim": claim_text,
                        "status": status,
                        "supported": supported,
                        "supporting_chunk_ids": sup_ids,
                        "explanation": ev.get("explanation") or ev.get("reasoning", ""),
                    })
                else:
                    # Deterministic heuristic fallback
                    status, supported, sup_ids, expl = self._heuristic_claim_check(claim_text, cited_ids, chunk_map)
                    evaluations.append({
                        "claim_id": cid,
                        "claim": claim_text,
                        "status": status,
                        "supported": supported,
                        "supporting_chunk_ids": sup_ids,
                        "explanation": expl,
                    })

        return evaluations

    def _call_llm_for_verification(
        self,
        claims: list[dict[str, Any]],
        chunk_map: dict[str, dict[str, Any]],
        question: str,
    ) -> list[dict[str, Any]]:
        """Call Groq LLM to verify claims."""
        evidence_payload = [
            {"chunk_id": cid, "text": c.get("text", "")}
            for cid, c in chunk_map.items()
        ]
        claims_payload = [
            {
                "claim_id": c.get("claim_id"),
                "claim": c.get("claim"),
                "cited_chunk_ids": c.get("cited_chunk_ids", []),
            }
            for c in claims
        ]

        q_str = f"USER QUESTION:\n{question}\n\n" if question else ""
        prompt = (
            f"{q_str}"
            f"ACCEPTED EVIDENCE CHUNKS (ONLY GROUND TRUTH):\n{json.dumps(evidence_payload, ensure_ascii=False, indent=2)}\n\n"
            f"CLAIMS TO VERIFY:\n{json.dumps(claims_payload, ensure_ascii=False, indent=2)}\n\n"
            "Evaluate each claim against the accepted evidence. Determine status: SUPPORTED, PARTIALLY_SUPPORTED, or UNSUPPORTED. "
            "Return JSON matching the schema."
        )

        messages = [
            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        if hasattr(self.llm_client, "complete_json") and callable(self.llm_client.complete_json):
            try:
                res = self.llm_client.complete_json(messages, model=self.model)
                return res.get("claim_evaluations", [])
            except Exception as exc:
                logger.exception("Verifier complete_json failed: %s", exc)

        raw_client = getattr(self.llm_client, "client", self.llm_client)
        if hasattr(raw_client, "chat") and hasattr(raw_client.chat, "completions"):
            try:
                response = raw_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=600,
                )
                data = _parse_json(response.choices[0].message.content.strip())
                return data.get("claim_evaluations", [])
            except Exception as exc:
                logger.exception("Verifier LLM verification failed: %s", exc)

        return []

    def _detect_deterministic_conflict(
        self,
        claim_text: str,
        chunk_map: dict[str, dict[str, Any]],
    ) -> tuple[bool, str]:
        """Detect obvious numerical or factual contradiction between claim and evidence."""
        claim_nums = set(re.findall(r"\b\d+(?:\.\d+)?%?", claim_text))
        all_chunk_text = " ".join(c.get("text", "") for c in chunk_map.values())
        evidence_nums = set(re.findall(r"\b\d+(?:\.\d+)?%?", all_chunk_text))

        # Direct percentage conflict check
        claim_pcts = {n for n in claim_nums if n.endswith("%")}
        evidence_pcts = {n for n in evidence_nums if n.endswith("%")}
        if claim_pcts and evidence_pcts and not (claim_pcts & evidence_pcts):
            return True, f"Incorrect claim: Claim asserts {', '.join(claim_pcts)}, which contradicts accepted evidence ({', '.join(evidence_pcts)})."

        # Direct numerical conflict check
        pure_claim_nums = {n for n in claim_nums if not n.endswith("%") and len(n) <= 6}
        pure_evidence_nums = {n for n in evidence_nums if not n.endswith("%") and len(n) <= 6}
        if pure_claim_nums and pure_evidence_nums and ("accuracy" in claim_text.lower() or "score" in claim_text.lower()):
            if not (pure_claim_nums & pure_evidence_nums):
                return True, f"Incorrect claim: Claim asserts {', '.join(pure_claim_nums)}, which contradicts accepted evidence ({', '.join(pure_evidence_nums)})."

        return False, ""

    def _heuristic_claim_check(
        self,
        claim_text: str,
        cited_ids: list[str],
        chunk_map: dict[str, dict[str, Any]],
    ) -> tuple[str, bool, list[str], str]:
        """Fallback word-overlap check for test harnesses without active LLM."""
        claim_words = set(re.findall(r"\w+", claim_text.lower()))
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "and", "or", "in", "on", "of", "to", "for", "with", "it", "that", "this"}
        content_words = claim_words - stop_words
        if not content_words:
            return "UNSUPPORTED", False, [], "Claim lacks content words to verify."

        best_overlap = 0.0
        best_cid = ""

        # Test cited IDs first, or all valid chunks
        search_cids = [cid for cid in cited_ids if cid in chunk_map] or list(chunk_map.keys())
        for cid in search_cids:
            chunk = chunk_map.get(cid)
            if chunk:
                c_words = set(re.findall(r"\w+", chunk.get("text", "").lower()))
                overlap = len(content_words & c_words) / len(content_words)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_cid = cid

        if best_overlap >= 0.70:
            return "SUPPORTED", True, [best_cid], f"Directly substantiated by chunk {best_cid} (high semantic overlap)."
        elif best_overlap >= 0.35:
            return "PARTIALLY_SUPPORTED", False, [best_cid], f"Partially corroborated by chunk {best_cid}, but contains unverified elements."
        else:
            return "UNSUPPORTED", False, [], "Claim is not substantiated by any accepted evidence chunk in the current document."

    def _resolve_trust_score(self, trust_score: float | Mapping[str, Any] | None) -> float:
        if trust_score is None:
            return 1.0
        if isinstance(trust_score, (int, float)):
            return float(trust_score)
        if isinstance(trust_score, dict):
            return float(trust_score.get("overall_score") or trust_score.get("trust_score", 1.0))
        return 1.0


def _parse_json(raw: str) -> dict[str, Any]:
    content = raw.strip()
    if content.startswith("```"):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
        if match:
            content = match.group(1).strip()
    try:
        return json.loads(content)
    except Exception:
        return {}


_verifier_agent: VerifierAgent | None = None


def get_verifier_agent() -> VerifierAgent:
    global _verifier_agent
    if _verifier_agent is None:
        _verifier_agent = VerifierAgent()
    return _verifier_agent
