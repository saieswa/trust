"""Critic Agent for evaluating retrieved evidence in Trust-Aware RAG.

Evaluates evidence for:
1. Relevance to the user question
2. Evidence strength
3. Source quality
4. Potential outdated information
5. Whether the evidence supports the question
6. Cross-chunk contradiction detection
7. Strict document isolation (unrelated documents are rejected)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Mapping, Sequence

from app.agents.contradiction_detector import get_contradiction_detector
from app.database.metadata import store_contradictions
from app.llm.groq_client import GroqClientError, GroqLLMClient, get_groq_client

logger = logging.getLogger(__name__)

CRITIC_SYSTEM_PROMPT = """You are an expert Critic Agent in a Trust-Aware RAG (Retrieval-Augmented Generation) system.
Your responsibility is to critically evaluate retrieved evidence chunks against a user's question and verify whether they factually support the question, assess source quality, detect potential outdatedness, and identify any contradictions between chunks.

You must evaluate each chunk independently along these criteria:
1. relevance: boolean (true if the chunk directly addresses or provides meaningful context for the user question, false otherwise)
2. support_status: string ("supported", "unsupported", or "contradicted")
   - "supported": The chunk provides factual evidence that directly answers or supports an answer to the question.
   - "unsupported": The chunk does not provide factual support for answering the question (e.g., irrelevant, tangential, or inconclusive).
   - "contradicted": The chunk's factual statements are directly contradicted by another retrieved chunk.
3. quality_assessment: object containing:
   - evidence_strength: "high", "medium", "low", or "none" (clarity, specificity, quantitative backing, factual rigor)
   - source_quality: "high", "medium", or "low" (reputability, authoritative context, completeness)
   - potential_outdated: boolean (true if the text indicates outdated, obsolete, or superseded information)
   - details: concise summary of the quality assessment
4. contradiction_status: string ("none" if no conflict with other retrieved chunks, or "contradiction_detected: <brief reason>")
5. explanation: clear, concise explanation of the evaluation decisions (relevance, support, quality, contradiction)

Cross-Chunk Contradiction Analysis:
Compare all relevant retrieved chunks with each other. If two or more chunks make conflicting or mutually incompatible factual claims (e.g. conflicting dates, figures, definitions, names, or outcomes):
- Mark contradiction_status as "contradiction_detected: <details>" for the conflicting chunks.
- Add an entry to the top-level "contradictions" list specifying the conflicting chunk IDs, conflicting claims, and explanation.
If no chunks conflict, contradiction_status must be "none" and "contradictions" must be an empty list [].

You must return ONLY valid JSON matching this schema:
{
  "evaluations": [
    {
      "chunk_id": "<chunk_id>",
      "relevance": true|false,
      "support_status": "supported"|"unsupported"|"contradicted",
      "quality_assessment": {
        "evidence_strength": "high"|"medium"|"low"|"none",
        "source_quality": "high"|"medium"|"low",
        "potential_outdated": true|false,
        "details": "<string>"
      },
      "contradiction_status": "none"|"contradiction_detected: <details>",
      "explanation": "<string>"
    }
  ],
  "contradictions": [
    {
      "chunk_id_a": "<chunk_id>",
      "chunk_id_b": "<chunk_id>",
      "claim_a": "<string>",
      "claim_b": "<string>",
      "explanation": "<string>"
    }
  ]
}
"""


class CriticAgent:
    """Agent responsible for evaluating retrieved evidence chunks and detecting contradictions."""

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

    def evaluate(
        self,
        question: str,
        evidence: Sequence[Mapping[str, Any] | Any],
        document_id: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate retrieved evidence chunks for relevance, quality, support, and contradictions.

        Strictly enforces document isolation: chunks from unrelated documents are rejected.
        """
        if not question or not question.strip():
            raise ValueError("question must contain non-empty text")

        # Normalize evidence items into standard dictionaries
        normalized_chunks: list[dict[str, Any]] = []
        for item in evidence:
            if hasattr(item, "__dict__"):
                chunk_dict = dict(item.__dict__)
            elif isinstance(item, Mapping):
                chunk_dict = dict(item)
            else:
                chunk_dict = {"text": str(item)}
            normalized_chunks.append(chunk_dict)

        if not normalized_chunks:
            return {
                "document_id": document_id or "",
                "evaluations": [],
                "contradictions": [],
                "summary": {
                    "total_chunks": 0,
                    "relevant_chunks": 0,
                    "supporting_chunks": 0,
                    "contradicting_chunks": 0,
                    "unrelated_chunks": 0,
                    "has_contradictions": False,
                },
            }

        # Resolve target document_id
        target_doc_id = (document_id or "").strip()
        if not target_doc_id:
            # If not provided, infer from first chunk if present
            target_doc_id = str(normalized_chunks[0].get("document_id") or "").strip()

        # Enforce document isolation
        valid_chunks: list[dict[str, Any]] = []
        unrelated_evaluations: list[dict[str, Any]] = []

        for chunk in normalized_chunks:
            c_doc_id = str(chunk.get("document_id") or "").strip()
            c_id = str(chunk.get("chunk_id") or f"chunk-{len(valid_chunks) + len(unrelated_evaluations)}")
            
            # Check document mismatch
            if target_doc_id and c_doc_id and c_doc_id != target_doc_id:
                logger.warning(
                    "Critic rejected cross-document evidence: chunk %s belongs to %s, expected %s",
                    c_id,
                    c_doc_id,
                    target_doc_id,
                )
                quality = {
                    "evidence_strength": "none",
                    "source_quality": "low",
                    "potential_outdated": False,
                    "valid_document": False,
                    "details": f"Rejected: chunk belongs to unrelated document '{c_doc_id}' instead of '{target_doc_id}'.",
                }
                unrelated_item = {
                    "document_id": c_doc_id,
                    "chunk_id": c_id,
                    "relevance": False,
                    "support_status": "unrelated_document",
                    "support status": "unrelated_document",
                    "quality_assessment": quality,
                    "quality assessment": quality,
                    "contradiction_status": "none",
                    "contradiction status": "none",
                    "explanation": f"Rejected: Chunk belongs to unrelated document '{c_doc_id}' instead of target document '{target_doc_id}'. Cross-document evidence is strictly prohibited.",
                }
                unrelated_evaluations.append(unrelated_item)
            else:
                valid_chunks.append({
                    "document_id": c_doc_id or target_doc_id,
                    "chunk_id": c_id,
                    "filename": chunk.get("filename", ""),
                    "page_number": chunk.get("page_number"),
                    "source": chunk.get("source", ""),
                    "text": chunk.get("text", ""),
                })

        valid_evaluations: list[dict[str, Any]] = []
        contradictions: list[dict[str, Any]] = []

        if valid_chunks:
            # 1. Run explicit contradiction detector (fast deterministic pass; full semantic check handled in Critic LLM call)
            try:
                detector = get_contradiction_detector()
                detector_contradictions = detector.detect_contradictions(valid_chunks, allow_semantic=False)
            except Exception as d_exc:
                logger.warning("Contradiction detector failed: %s", d_exc)
                detector_contradictions = []

            # 2. Run LLM evaluation for chunk quality, relevance, support
            llm_result = self._evaluate_with_llm(question, valid_chunks)
            raw_evals = llm_result.get("evaluations", [])
            raw_contradictions = llm_result.get("contradictions", [])

            # Merge contradictions cleanly, deduplicating pairs
            seen_pairs = set()
            merged_contradictions: list[dict[str, Any]] = []
            for c in detector_contradictions + raw_contradictions:
                id_a = str(c.get("chunk_a") or c.get("chunk_id_a") or "")
                id_b = str(c.get("chunk_b") or c.get("chunk_id_b") or "")
                pair_key = (min(id_a, id_b), max(id_a, id_b))
                if pair_key not in seen_pairs and id_a != id_b:
                    seen_pairs.add(pair_key)
                    c_norm = dict(c)
                    c_norm["contradiction"] = True
                    c_norm["chunk_a"] = id_a
                    c_norm["chunk_b"] = id_b
                    c_norm["chunk_id_a"] = id_a
                    c_norm["chunk_id_b"] = id_b
                    c_norm["reason"] = c.get("reason") or c.get("explanation") or "Factual contradiction detected."
                    c_norm["explanation"] = c_norm["reason"]
                    c_norm["severity"] = c.get("severity", "high")
                    merged_contradictions.append(c_norm)

            contradictions = merged_contradictions

            # Index LLM evaluations by chunk_id
            eval_by_id: dict[str, dict[str, Any]] = {}
            for e in raw_evals:
                cid = str(e.get("chunk_id", ""))
                if cid:
                    eval_by_id[cid] = e

            for chunk in valid_chunks:
                cid = chunk["chunk_id"]
                doc_id = chunk["document_id"]
                evaluated = eval_by_id.get(cid)

                if evaluated:
                    rel = bool(evaluated.get("relevance", False))
                    supp = str(evaluated.get("support_status", "unsupported")).lower()
                    if supp not in ("supported", "unsupported", "contradicted"):
                        supp = "supported" if rel else "unsupported"
                    
                    raw_quality = evaluated.get("quality_assessment") or {}
                    if not isinstance(raw_quality, dict):
                        raw_quality = {"details": str(raw_quality)}
                    
                    quality = {
                        "evidence_strength": str(raw_quality.get("evidence_strength", "medium")),
                        "source_quality": str(raw_quality.get("source_quality", "medium")),
                        "potential_outdated": bool(raw_quality.get("potential_outdated", False)),
                        "details": str(raw_quality.get("details", "")),
                    }
                    contra_status = str(evaluated.get("contradiction_status", "none"))
                    explanation = str(evaluated.get("explanation", ""))
                else:
                    # Fallback if specific chunk was missed in LLM array
                    rel = False
                    supp = "unsupported"
                    quality = {
                        "evidence_strength": "low",
                        "source_quality": "low",
                        "potential_outdated": False,
                        "details": "Evaluation unavailable from LLM response.",
                    }
                    contra_status = "none"
                    explanation = "Chunk could not be evaluated by LLM."

                # Check if this chunk is part of detected contradictions
                for c in contradictions:
                    if cid in (str(c.get("chunk_a")), str(c.get("chunk_b")), str(c.get("chunk_id_a")), str(c.get("chunk_id_b"))):
                        reason_msg = c.get("reason") or c.get("explanation") or "Conflicting claims"
                        contra_status = f"contradiction_detected: {reason_msg}"
                        supp = "contradicted"

                item = {
                    "document_id": doc_id,
                    "chunk_id": cid,
                    "relevance": rel,
                    "support_status": supp,
                    "support status": supp,
                    "quality_assessment": quality,
                    "quality assessment": quality,
                    "contradiction_status": contra_status,
                    "contradiction status": contra_status,
                    "explanation": explanation,
                }
                valid_evaluations.append(item)

            if target_doc_id and contradictions:
                try:
                    store_contradictions(target_doc_id, contradictions, query=question)
                except Exception as meta_exc:
                    logger.warning("Failed to persist contradictions for document %s: %s", target_doc_id, meta_exc)

        # Merge valid evaluations and unrelated evaluations
        all_evaluations = valid_evaluations + unrelated_evaluations

        # Compute summary
        total_chunks = len(all_evaluations)
        relevant_chunks = sum(1 for e in all_evaluations if e["relevance"])
        supporting_chunks = sum(1 for e in all_evaluations if e["support_status"] == "supported")
        contradicting_chunks = sum(
            1 for e in all_evaluations
            if "contradiction" in e["contradiction_status"].lower() or e["support_status"] == "contradicted"
        )
        unrelated_chunks = len(unrelated_evaluations)

        return {
            "document_id": target_doc_id,
            "evaluations": all_evaluations,
            "contradictions": contradictions,
            "summary": {
                "total_chunks": total_chunks,
                "relevant_chunks": relevant_chunks,
                "supporting_chunks": supporting_chunks,
                "contradicting_chunks": contradicting_chunks,
                "unrelated_chunks": unrelated_chunks,
                "has_contradictions": len(contradictions) > 0 or contradicting_chunks > 0,
            },
        }

    def _evaluate_with_llm(
        self,
        question: str,
        valid_chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Invoke Groq/Llama via JSON mode and parse structured evaluation result."""
        evidence_payload = [
            {
                "chunk_id": c["chunk_id"],
                "document_id": c["document_id"],
                "filename": c.get("filename"),
                "page_number": c.get("page_number"),
                "source": c.get("source"),
                "text": c.get("text", ""),
            }
            for c in valid_chunks
        ]

        prompt = (
            f"USER QUESTION:\n{question}\n\n"
            f"RETRIEVED EVIDENCE CHUNKS:\n{json.dumps(evidence_payload, ensure_ascii=False, indent=2)}\n\n"
            "Evaluate each chunk thoroughly for relevance, evidence strength, source quality, outdated information, "
            "and whether it supports the question. Compare chunks to detect any factual contradictions. Return the evaluation JSON."
        )

        messages = [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # Use complete_json if available on the client
        if hasattr(self.llm_client, "complete_json") and callable(self.llm_client.complete_json):
            try:
                return self.llm_client.complete_json(messages, model=self.model)
            except Exception as exc:
                logger.exception("Critic complete_json failed: %s", exc)
                raise GroqClientError(f"Critic evaluation failed: {type(exc).__name__}") from exc

        # Or invoke Groq client directly
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
                raw_content = response.choices[0].message.content.strip()
                return _parse_json_response(raw_content)
            except Exception as exc:
                logger.exception("Groq chat completion failed in Critic: %s", exc)
                raise GroqClientError(f"Critic LLM evaluation failed: {type(exc).__name__}: {exc}") from exc

        # Fallback if mock client doesn't support chat completions
        logger.warning("LLM client does not support chat completions; using fallback.")
        return {"evaluations": [], "contradictions": []}


def _parse_json_response(raw_content: str) -> dict[str, Any]:
    """Parse JSON string, stripping markdown code blocks if present."""
    content = raw_content.strip()
    if content.startswith("```"):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
        if match:
            content = match.group(1).strip()
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return data
        return {"evaluations": []}
    except json.JSONDecodeError as exc:
        logger.warning("Failed to decode JSON from LLM: %s. Content: %s", exc, raw_content[:200])
        return {"evaluations": []}


_critic_agent: CriticAgent | None = None


def get_critic_agent() -> CriticAgent:
    """Return process-wide singleton CriticAgent instance."""
    global _critic_agent
    if _critic_agent is None:
        _critic_agent = CriticAgent()
    return _critic_agent
