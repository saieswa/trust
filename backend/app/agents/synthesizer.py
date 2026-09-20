"""Synthesizer Agent for Trust-Aware RAG.

Synthesizes grounded answers strictly from Critic-approved, non-contradictory
evidence chunks belonging exclusively to the target document.
Every factual claim is linked to specific source citations (chunk IDs and page numbers).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Mapping, Sequence

from app.llm.groq_client import GroqClientError, GroqLLMClient, get_groq_client, sanitize_answer

logger = logging.getLogger(__name__)

SYNTHESIZER_SYSTEM_PROMPT = """You are an expert Synthesizer Agent in a Trust-Aware RAG system.
Your mission is to formulate an accurate, concise, readable, and strictly evidence-grounded answer to the user's question using ONLY from accepted evidence chunks provided below.

STRICT INSTRUCTIONS & REQUIREMENTS:
1. Answer ONLY from accepted evidence provided in the prompt context. Directly answer the question first in clear, natural, and grammatically complete sentences.
2. Use ONLY information strictly supported by the accepted evidence. Do NOT invent facts or extrapolate beyond what the evidence directly states.
3. Do NOT fill missing information using outside or pre-trained knowledge.
4. Do NOT cite sources or documents not present in the provided evidence.
5. Keep the answer readable, well-structured, clear, and understandable to a student.
6. Clearly indicate uncertainty when evidence is incomplete or when the answer is not present in the document:
   - If the answer cannot be found in the document evidence, explicitly state: "Based on the provided document evidence, there is no information to answer this question."
   - If the evidence only partially answers the question, answer the supported parts and clearly explain what information is missing.
7. Do NOT copy incomplete sentence fragments, raw slice tags, or sliced text.
8. Do NOT combine unrelated evidence.
9. Do NOT mention internal chunk IDs, UUIDs, or database identifiers anywhere in the text of the answer.
10. Do NOT mention the retrieval or evaluation process unless specifically asked.
11. Prefer a clear, concise, well-structured explanation over blindly reproducing raw evidence.
12. Ensure the final answer is grammatically complete and coherent.

STRUCTURE & FORMATTING REQUIREMENTS:
- Always format the answer with clear markdown headings (###) and bullet points where helpful.
- Provide a clear 1-2 sentence core definition under an introductory heading (e.g., "### Overview").
- Group important concepts under distinct headings (e.g., "### Core Principles", "### How It Works", "### Key Conditions", "### Applications").
- Conclude with a brief 1-sentence summary under "### Summary".
- If citing sources in the answer text, use ONLY readable human citations such as: "(Source: <filename>, Page <page_number>)".

You must return ONLY valid JSON matching this schema:
{
  "answer": "Clear, readable markdown answer with structured headings and bullet points strictly from accepted evidence",
  "claims": [
    {
      "claim_id": "claim-1",
      "text": "<specific atomic factual statement>",
      "cited_chunk_ids": ["<chunk_id>"]
    }
  ]
}
"""

REVISER_SYSTEM_PROMPT = """You are an expert Reviser Agent in a Trust-Aware RAG system.
Your mission is to perform a controlled revision of a draft answer to remove, correct, or refine statements that were flagged as UNSUPPORTED by the Verifier.

STRICT REVISION RULES:
1. Revise the draft answer so that every statement is 100% grounded in the accepted evidence chunks below.
2. Remove or correct any claim flagged as unsupported if it cannot be directly substantiated by the accepted evidence.
3. Do NOT invent facts or extrapolate beyond what the accepted evidence directly states.
4. Do NOT introduce outside, general, or pre-trained knowledge.
5. If removing unsupported claims leaves the answer incomplete, explicitly declare what the document states and acknowledge what is missing.
6. Keep the revised answer clear, readable, and concise.
7. Extract the atomic claims made in your revised answer and tie them to cited chunk IDs.
8. CITATIONS & SOURCE FORMATTING:
   - NEVER output internal IDs, UUIDs, chunk keys, or database IDs in the text of the answer.
   - Do NOT write citations like {chunk_id}, (cite: chunk_id), or [chunk-0001].
   - If citing sources in the answer text, use ONLY readable human citations such as: "(Source: <filename>, Page <page_number>)".

You must return ONLY valid JSON matching this schema:
{
  "answer": "Clear, readable markdown revised answer strictly from accepted evidence",
  "claims": [
    {
      "claim_id": "claim-1",
      "text": "<specific factual statement in revised answer>",
      "cited_chunk_ids": ["<chunk_id>"]
    }
  ]
}
"""


class SynthesizerAgent:
    """Agent responsible for synthesizing answers grounded strictly on Critic-approved evidence."""

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
        return getattr(self.llm_client, "model", "qwen/qwen3.8-27b")

    def synthesize(
        self,
        question: str,
        evidence: Sequence[Mapping[str, Any] | Any],
        document_id: str,
        metadata: Mapping[str, Any] | None = None,
        trust_decision: Any = None,
        evaluations: Sequence[Mapping[str, Any]] | None = None,
        contradictions: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Synthesize an answer using ONLY Critic-approved evidence and document metadata."""
        if not question or not question.strip():
            raise ValueError("question must contain non-empty text")
        if not document_id or not document_id.strip():
            raise ValueError("document_id must contain non-empty text")

        doc_id = document_id.strip()

        # Step 1: Filter evidence strictly:
        # - Exclude rejected chunks
        # - Exclude chunks from other/unrelated documents
        # - Exclude unmarked contradictory evidence
        valid_evidence = self._filter_evidence(
            evidence=evidence,
            target_document_id=doc_id,
            evaluations=evaluations,
            contradictions=contradictions,
        )

        decision_dict: dict[str, Any] = {}
        if trust_decision is not None:
            if hasattr(trust_decision, "to_dict"):
                decision_dict = trust_decision.to_dict()
            elif isinstance(trust_decision, dict):
                decision_dict = dict(trust_decision)

        # If no valid critic-approved evidence remains
        if not valid_evidence:
            return {
                "answer": "I don't have enough reliable evidence in the uploaded document to answer this question.",
                "cited_evidence": [],
                "source_information": [],
                "claims": [],
                "sources": [],
                "document_id": doc_id,
                "trust_decision": decision_dict,
                "metadata": dict(metadata or {}),
            }

        # Step 2: Extract source information
        source_info = self._extract_sources(valid_evidence, doc_id)

        # Step 3: Call LLM for synthesis and claim extraction
        llm_output = self._call_llm(
            question=question.strip(),
            valid_evidence=valid_evidence,
            document_id=doc_id,
            metadata=metadata,
            trust_decision=decision_dict,
        )

        raw_answer = str(llm_output.get("answer", "")).strip()
        has_explicit_claims_key = "claims" in llm_output and isinstance(llm_output.get("claims"), list)
        raw_claims = llm_output.get("claims", [])

        claims: list[dict[str, Any]] = []
        for idx, c in enumerate(raw_claims):
            if isinstance(c, dict) and c.get("text"):
                c_id = c.get("claim_id") or f"claim-{idx + 1}"
                chunk_ids = [str(cid) for cid in c.get("cited_chunk_ids", [])]
                claims.append({
                    "claim_id": c_id,
                    "text": c["text"],
                    "cited_chunk_ids": chunk_ids,
                })

        # Fallback sentence-level extraction only if LLM output lacked claims field
        # and answer is not expressing absence of evidence or uncertainty
        is_uncertainty_or_absence = any(
            phrase in raw_answer.lower()
            for phrase in [
                "no information",
                "not enough reliable evidence",
                "not mentioned",
                "does not mention",
                "no evidence",
                "cannot be answered",
                "does not contain",
            ]
        )
        if not has_explicit_claims_key and not claims and raw_answer and not is_uncertainty_or_absence:
            claims = self._fallback_claims_extraction(raw_answer, valid_evidence)

        # Step 4: Map cited evidence chunks
        cited_ids = set()
        for clm in claims:
            for cid in clm.get("cited_chunk_ids", []):
                cited_ids.add(str(cid))

        cited_evidence: list[dict[str, Any]] = []
        for c in valid_evidence:
            cid = str(c.get("chunk_id", ""))
            # If specific citations exist, filter to cited ones; otherwise include all valid chunks
            if not cited_ids or cid in cited_ids:
                cited_evidence.append({
                    "chunk_id": cid,
                    "text": c.get("text", ""),
                    "source": c.get("source") or c.get("filename"),
                    "page_number": c.get("page_number"),
                    "document_id": c.get("document_id", doc_id),
                    "has_contradiction": c.get("has_contradiction", False),
                    "contradiction_reason": c.get("contradiction_reason"),
                })

        final_answer = raw_answer or "I don't have enough reliable evidence in the uploaded document to answer this question."
        final_answer = sanitize_answer(final_answer, evidence=valid_evidence)

        return {
            "answer": final_answer,
            "cited_evidence": cited_evidence,
            "source_information": source_info,
            "claims": claims,
            "sources": source_info,  # Backwards compatibility
            "document_id": doc_id,
            "trust_decision": decision_dict,
            "metadata": dict(metadata or {}),
        }

    def revise(
        self,
        question: str,
        draft_answer: str,
        unsupported_claims: Sequence[Mapping[str, Any]],
        evidence: Sequence[Mapping[str, Any] | Any],
        document_id: str,
        metadata: Mapping[str, Any] | None = None,
        trust_decision: Any = None,
    ) -> dict[str, Any]:
        """Perform one controlled revision of the draft answer to remove/correct unsupported claims.

        Strictly uses ONLY accepted evidence chunks. Does NOT introduce outside knowledge.
        """
        doc_id = document_id.strip() if document_id else ""
        valid_evidence = self._filter_evidence(evidence=evidence, target_document_id=doc_id)

        decision_dict: dict[str, Any] = {}
        if trust_decision is not None:
            if hasattr(trust_decision, "to_dict"):
                decision_dict = trust_decision.to_dict()
            elif isinstance(trust_decision, dict):
                decision_dict = dict(trust_decision)

        if not valid_evidence:
            return {
                "answer": "I don't have enough reliable evidence in the uploaded document to answer this question.",
                "cited_evidence": [],
                "source_information": [],
                "claims": [],
                "sources": [],
                "document_id": doc_id,
                "trust_decision": decision_dict,
                "metadata": dict(metadata or {}),
            }

        unsupported_texts: list[str] = []
        for uc in unsupported_claims:
            uc_map = dict(uc.__dict__) if hasattr(uc, "__dict__") else dict(uc)
            t = uc_map.get("claim") or uc_map.get("text", "")
            if t:
                unsupported_texts.append(t)

        source_info = self._extract_sources(valid_evidence, doc_id)

        llm_output = self._call_llm_for_revision(
            question=question,
            draft_answer=draft_answer,
            unsupported_texts=unsupported_texts,
            valid_evidence=valid_evidence,
            document_id=doc_id,
            metadata=metadata,
            trust_decision=decision_dict,
        )

        raw_answer = str(llm_output.get("answer", "")).strip()
        has_explicit_claims_key = "claims" in llm_output and isinstance(llm_output.get("claims"), list)
        raw_claims = llm_output.get("claims", [])

        claims: list[dict[str, Any]] = []
        for idx, c in enumerate(raw_claims):
            if isinstance(c, dict) and c.get("text"):
                c_id = c.get("claim_id") or f"claim-{idx + 1}"
                chunk_ids = [str(cid) for cid in c.get("cited_chunk_ids", [])]
                claims.append({
                    "claim_id": c_id,
                    "text": c["text"],
                    "cited_chunk_ids": chunk_ids,
                })

        if not has_explicit_claims_key and not claims and raw_answer:
            claims = self._fallback_claims_extraction(raw_answer, valid_evidence)

        cited_ids = set()
        for clm in claims:
            for cid in clm.get("cited_chunk_ids", []):
                cited_ids.add(str(cid))

        cited_evidence: list[dict[str, Any]] = []
        for c in valid_evidence:
            cid = str(c.get("chunk_id", ""))
            if not cited_ids or cid in cited_ids:
                cited_evidence.append({
                    "chunk_id": cid,
                    "text": c.get("text", ""),
                    "source": c.get("source") or c.get("filename"),
                    "page_number": c.get("page_number"),
                    "document_id": c.get("document_id", doc_id),
                })

        final_answer = raw_answer or draft_answer
        final_answer = sanitize_answer(final_answer, evidence=valid_evidence)
        return {
            "answer": final_answer,
            "cited_evidence": cited_evidence,
            "source_information": source_info,
            "claims": claims,
            "sources": source_info,
            "document_id": doc_id,
            "trust_decision": decision_dict,
            "metadata": dict(metadata or {}),
        }

    def _call_llm_for_revision(
        self,
        question: str,
        draft_answer: str,
        unsupported_texts: list[str],
        valid_evidence: list[dict[str, Any]],
        document_id: str,
        metadata: Mapping[str, Any] | None = None,
        trust_decision: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        short_to_real_id: dict[str, str] = {}
        evidence_payload = []
        for idx, c in enumerate(valid_evidence):
            raw_cid = str(c.get("chunk_id") or "")
            clean_cid = raw_cid.split("::")[-1] if "::" in raw_cid else (raw_cid or f"chunk-{idx + 1}")
            short_to_real_id[clean_cid] = raw_cid
            if raw_cid:
                short_to_real_id[raw_cid] = raw_cid

            evidence_payload.append({
                "chunk_id": clean_cid,
                "page_number": c.get("page_number"),
                "filename": c.get("filename"),
                "source": c.get("source"),
                "text": c.get("text", ""),
            })

        prompt = (
            f"USER QUESTION:\n{question}\n\n"
            f"DRAFT ANSWER REQUIRING REVISION:\n{draft_answer}\n\n"
            f"UNSUPPORTED CLAIMS IDENTIFIED BY VERIFIER (MUST BE REMOVED OR CORRECTED):\n{json.dumps(unsupported_texts, ensure_ascii=False, indent=2)}\n\n"
            f"ACCEPTED EVIDENCE CHUNKS (ONLY GROUND TRUTH ALLOWED):\n{json.dumps(evidence_payload, ensure_ascii=False, indent=2)}\n\n"
            "Revise the draft answer strictly using ONLY the accepted evidence chunks above. "
            "Do NOT introduce any outside or pre-trained knowledge. Return valid JSON."
        )

        messages = [
            {"role": "system", "content": REVISER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        def _remap_claims(result_dict: dict[str, Any]) -> dict[str, Any]:
            if isinstance(result_dict, dict) and "claims" in result_dict and isinstance(result_dict["claims"], list):
                for claim in result_dict["claims"]:
                    if isinstance(claim, dict) and "cited_chunk_ids" in claim:
                        raw_cids = [str(cid) for cid in claim.get("cited_chunk_ids", [])]
                        claim["cited_chunk_ids"] = [short_to_real_id.get(cid, cid) for cid in raw_cids]
            return result_dict

        if hasattr(self.llm_client, "complete_json") and callable(self.llm_client.complete_json):
            try:
                res = self.llm_client.complete_json(messages, model=self.model)
                return _remap_claims(res)
            except Exception as exc:
                logger.exception("Synthesizer revise complete_json failed: %s", exc)

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
                content = response.choices[0].message.content.strip()
                return _remap_claims(_parse_json(content))
            except Exception as exc:
                logger.exception("Synthesizer LLM revision failed: %s", exc)
                raise GroqClientError(f"Synthesizer LLM revision failed: {exc}") from exc

        # Deterministic heuristic fallback when LLM is mocked
        revised_sentences = []
        for sent in re.split(r"(?<=[.!?])\s+", draft_answer):
            sent_clean = sent.strip()
            if not sent_clean:
                continue
            is_unsupported = any(
                ut.lower() in sent_clean.lower() or sent_clean.lower() in ut.lower()
                for ut in unsupported_texts
            )
            if not is_unsupported:
                revised_sentences.append(sent_clean)

        revised_text = " ".join(revised_sentences).strip()
        if not revised_text:
            revised_text = f"Based on the provided document evidence, {valid_evidence[0].get('text', '')[:100]}."

        return {
            "answer": revised_text,
            "claims": [
                {
                    "claim_id": "claim-1",
                    "text": revised_text[:120],
                    "cited_chunk_ids": [str(valid_evidence[0].get("chunk_id", ""))],
                }
            ],
        }

    def _filter_evidence(
        self,
        evidence: Sequence[Mapping[str, Any] | Any],
        target_document_id: str,
        evaluations: Sequence[Mapping[str, Any]] | None = None,
        contradictions: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Filter out rejected chunks, unrelated documents, and unmarked contradictory evidence."""
        # Build evaluation lookup
        eval_by_id: dict[str, Mapping[str, Any]] = {}
        if evaluations:
            for ev in evaluations:
                ev_map = dict(ev.__dict__) if hasattr(ev, "__dict__") else dict(ev)
                cid = str(ev_map.get("chunk_id", ""))
                if cid:
                    eval_by_id[cid] = ev_map

        # Build contradiction lookup
        contradictory_chunk_ids: set[str] = set()
        contra_reasons: dict[str, str] = {}
        if contradictions:
            for cont in contradictions:
                cont_map = dict(cont.__dict__) if hasattr(cont, "__dict__") else dict(cont)
                reason = cont_map.get("reason") or cont_map.get("explanation") or "Factual conflict detected."
                for key in ("chunk_a", "chunk_b", "chunk_id_a", "chunk_id_b"):
                    val = cont_map.get(key)
                    if val:
                        contradictory_chunk_ids.add(str(val))
                        contra_reasons[str(val)] = reason

        valid_chunks: list[dict[str, Any]] = []

        for item in evidence:
            chunk = dict(item.__dict__) if hasattr(item, "__dict__") else dict(item)
            cid = str(chunk.get("chunk_id", "")).strip()
            c_doc = str(chunk.get("document_id", "")).strip()

            # Rule 1: Exclude unrelated documents & cross-document leakage
            if c_doc and c_doc != target_document_id:
                logger.warning("Synthesizer rejected chunk %s: document mismatch (%s != %s)", cid, c_doc, target_document_id)
                continue
            if chunk.get("support_status") == "unrelated_document":
                logger.warning("Synthesizer rejected chunk %s: marked unrelated_document", cid)
                continue

            # Rule 2: Exclude rejected chunks from evaluations or chunk flags
            if chunk.get("is_rejected") or chunk.get("rejected"):
                logger.warning("Synthesizer rejected chunk %s: marked rejected on chunk", cid)
                continue
            if chunk.get("relevance") is False:
                logger.warning("Synthesizer rejected chunk %s: chunk relevance is False", cid)
                continue
            if chunk.get("support_status") in ("rejected", "unsupported"):
                logger.warning("Synthesizer rejected chunk %s: chunk support_status is %s", cid, chunk.get("support_status"))
                continue

            if cid in eval_by_id:
                ev = eval_by_id[cid]
                if not ev.get("relevance", True):
                    logger.warning("Synthesizer rejected chunk %s: Critic marked relevance=False", cid)
                    continue
                if ev.get("support_status") in ("unrelated_document", "rejected"):
                    logger.warning("Synthesizer rejected chunk %s: Critic marked support_status=%s", cid, ev.get("support_status"))
                    continue
                if ev.get("is_rejected"):
                    logger.warning("Synthesizer rejected chunk %s: Critic marked is_rejected=True", cid)
                    continue

            # Rule 3: Contradictory evidence handling
            # Must NOT receive contradictory evidence unless explicitly marked
            is_in_contradiction = (cid in contradictory_chunk_ids) or chunk.get("has_contradiction", False)
            if is_in_contradiction:
                # Check if it is explicitly marked
                is_marked = bool(chunk.get("has_contradiction") or chunk.get("contradiction_marked") or chunk.get("contradiction_reason"))
                if not is_marked and cid in contradictory_chunk_ids:
                    # Unmarked contradiction: MUST NOT receive it
                    logger.warning("Synthesizer excluded unmarked contradictory chunk %s", cid)
                    continue

                # If marked, annotate the chunk with explicit notice
                chunk["has_contradiction"] = True
                if not chunk.get("contradiction_reason") and cid in contra_reasons:
                    chunk["contradiction_reason"] = contra_reasons[cid]

            valid_chunks.append(chunk)

        return valid_chunks

    def _call_llm(
        self,
        question: str,
        valid_evidence: list[dict[str, Any]],
        document_id: str,
        metadata: Mapping[str, Any] | None = None,
        trust_decision: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        short_to_real_id: dict[str, str] = {}
        evidence_payload = []
        for idx, c in enumerate(valid_evidence):
            raw_cid = str(c.get("chunk_id") or "")
            clean_cid = raw_cid.split("::")[-1] if "::" in raw_cid else (raw_cid or f"chunk-{idx + 1}")
            short_to_real_id[clean_cid] = raw_cid
            if raw_cid:
                short_to_real_id[raw_cid] = raw_cid

            entry = {
                "chunk_id": clean_cid,
                "page_number": c.get("page_number"),
                "filename": c.get("filename"),
                "source": c.get("source"),
                "text": c.get("text", ""),
            }
            if c.get("has_contradiction"):
                entry["NOTE"] = f"CONTRADICTION DETECTED: {c.get('contradiction_reason', 'Conflicting evidence observed.')}"
            evidence_payload.append(entry)

        meta_section = ""
        if metadata:
            meta_section = f"DOCUMENT METADATA:\n{json.dumps(dict(metadata), ensure_ascii=False, indent=2)}\n\n"

        decision_section = ""
        if trust_decision:
            decision_section = f"TRUST DECISION:\n{json.dumps(trust_decision, ensure_ascii=False, indent=2)}\n\n"

        prompt = (
            f"USER QUESTION:\n{question}\n\n"
            f"{meta_section}"
            f"{decision_section}"
            f"CRITIC-APPROVED EVIDENCE CHUNKS (ONLY ACCEPTED EVIDENCE):\n{json.dumps(evidence_payload, ensure_ascii=False, indent=2)}\n\n"
            "Formulate a structured, readable answer grounded ONLY in the above accepted evidence and extract every factual claim with its cited chunk IDs.\n"
            "If the question cannot be answered from the provided evidence, explicitly declare the absence of evidence and express uncertainty. "
            "Return valid JSON matching the schema."
        )

        messages = [
            {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        def _remap_claims(result_dict: dict[str, Any]) -> dict[str, Any]:
            if isinstance(result_dict, dict) and "claims" in result_dict and isinstance(result_dict["claims"], list):
                for claim in result_dict["claims"]:
                    if isinstance(claim, dict) and "cited_chunk_ids" in claim:
                        raw_cids = [str(cid) for cid in claim.get("cited_chunk_ids", [])]
                        claim["cited_chunk_ids"] = [short_to_real_id.get(cid, cid) for cid in raw_cids]
            return result_dict

        if hasattr(self.llm_client, "complete_json") and callable(self.llm_client.complete_json):
            try:
                res = self.llm_client.complete_json(messages, model=self.model)
                return _remap_claims(res)
            except Exception as exc:
                logger.exception("Synthesizer complete_json failed: %s", exc)

        raw_client = getattr(self.llm_client, "client", self.llm_client)
        if hasattr(raw_client, "chat") and hasattr(raw_client.chat, "completions"):
            try:
                response = raw_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=1024,
                )
                content = response.choices[0].message.content.strip()
                return _remap_claims(_parse_json(content))
            except Exception as exc:
                logger.warning("Synthesizer LLM generation failed: %s, using deterministic fallback", exc)

        # Deterministic fallback when LLM is unavailable / rate-limited:
        # Extract complete, grammatical sentences matching the question keywords.
        # NEVER return a sliced character fragment prefixed with 'Answer grounded on evidence:'.
        q_lower = question.lower()
        q_keywords = [
            w for w in re.findall(r"\w+", q_lower)
            if len(w) > 2 and w not in ("what", "when", "where", "which", "does", "have", "with", "this", "that", "from", "your", "explain")
        ]

        candidate_sentences: list[tuple[str, str]] = []
        for chunk in valid_evidence:
            cid = str(chunk.get("chunk_id", ""))
            c_text = chunk.get("text", "")
            # Split into complete sentences
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", c_text) if len(s.strip()) > 20]
            for s in sents:
                s_lower = s.lower()
                matches = sum(1 for kw in q_keywords if kw in s_lower)
                if matches > 0:
                    candidate_sentences.append((s, cid))

        if not candidate_sentences:
            return {
                "answer": "I could not find enough information in the uploaded document to provide a reliable answer.",
                "claims": [],
            }

        selected = candidate_sentences[:3]
        clean_answer = " ".join(s[0] for s in selected)
        claims = [
            {
                "claim_id": f"claim-{idx+1}",
                "text": s[0],
                "cited_chunk_ids": [s[1]],
            }
            for idx, s in enumerate(selected)
        ]
        return {
            "answer": clean_answer,
            "claims": claims,
        }

    def _extract_sources(self, evidence: list[dict[str, Any]], document_id: str) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        seen = set()
        for e in evidence:
            raw_name = e.get("filename") or "Document"
            fname = raw_name.split("#")[0].strip()
            pnum = e.get("page") if e.get("page") is not None else e.get("page_number")
            canonical = f"{fname} — Page {pnum}" if pnum is not None else fname
            key = (fname, str(pnum) if pnum is not None else "")
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source": canonical,
                    "canonical_source": canonical,
                    "filename": fname,
                    "page_number": pnum,
                    "page": pnum,
                    "document_id": document_id,
                })
        return sources

    def _fallback_claims_extraction(
        self,
        answer: str,
        evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Extract sentence-level claims as fallback if JSON parsing lacked claims."""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 15]
        default_chunk_id = str(evidence[0].get("chunk_id", "")) if evidence else ""
        return [
            {
                "claim_id": f"claim-{i+1}",
                "text": s,
                "cited_chunk_ids": [default_chunk_id] if default_chunk_id else [],
            }
            for i, s in enumerate(sentences[:5])
        ]


def _parse_json(raw_content: str) -> dict[str, Any]:
    content = raw_content.strip()
    if content.startswith("```"):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
        if match:
            content = match.group(1).strip()
    try:
        return json.loads(content)
    except Exception:
        return {"answer": content, "claims": []}


_synthesizer_agent: SynthesizerAgent | None = None


def get_synthesizer_agent() -> SynthesizerAgent:
    global _synthesizer_agent
    if _synthesizer_agent is None:
        _synthesizer_agent = SynthesizerAgent()
    return _synthesizer_agent
