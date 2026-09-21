"""Explicit contradiction detection for retrieved evidence chunks in Trust-Aware RAG.

Detects when two retrieved evidence chunks disagree about the same claim.
Uses deterministic checks (numerical metrics, polar negations, topic filters)
to minimize LLM calls, and Groq/Llama for nuanced semantic contradiction analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import re
from typing import Any, Mapping, Sequence

from app.llm.groq_client import GroqClientError, GroqLLMClient, get_groq_client

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than",
    "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}

NUMERICAL_CLAIM_PATTERN = re.compile(
    r"\b(?:the\s+)?([a-zA-Z_][a-zA-Z0-9_\-\s]{1,30}?)\s+(?:is|was|are|were|reached|stands?\s+at|equals?|measured|initialized\s+to|strictly\s+set\s+to|set\s+to|of)\s+([~><=]?\s*\$?\d+(?:\.\d+)?\s*(?:%|percent\b|percentage\b|ms\b|s\b|sec\b|seconds\b|min\b|minutes\b|hours\b|days\b|years\b|km\b|m\b|kg\b|gb\b|mb\b|epochs\b)?)",
    re.IGNORECASE,
)

REVERSE_NUMERICAL_CLAIM_PATTERN = re.compile(
    r"([~><=]?\s*\$?\d+(?:\.\d+)?\s*(?:%|percent\b|percentage\b|ms\b|s\b|sec\b|min\b|hours\b|days\b|epochs\b))\s+([a-zA-Z_][a-zA-Z0-9_\-]{2,25})\b",
    re.IGNORECASE,
)

DISTINCT_MODIFIER_SETS = [
    {"train", "training", "test", "testing", "validation", "val"},
    {"min", "minimum", "max", "maximum"},
    {"initial", "final", "starting", "ending"},
    {"top-1", "top-5", "top1", "top5"},
    {"cpu", "gpu", "tpu"},
]

POLAR_NEGATIONS = [
    (r"\b(?:was\s+|is\s+)?approved\b", r"\b(?:was\s+|is\s+)?(?:rejected|disapproved|not\s+approved)\b"),
    (r"\b(?:increased|rose|grew)\b", r"\b(?:decreased|fell|dropped|declined)\b"),
    (r"\b(?:is\s+|was\s+)?enabled\b", r"\b(?:is\s+|was\s+)?(?:disabled|deactivated|not\s+enabled)\b"),
    (r"\b(?:is\s+|was\s+)?supported\b", r"\b(?:is\s+|was\s+)?(?:unsupported|not\s+supported)\b"),
    (r"\b(?:is\s+|was\s+)?possible\b", r"\b(?:is\s+|was\s+)?(?:impossible|infeasible|not\s+possible)\b"),
    (r"\b(?:was\s+|is\s+)?(?:a\s+)?success(?:ful)?\b", r"\b(?:was\s+|is\s+)?(?:a\s+)?(?:failure|failed|unsuccessful)\b"),
    (r"\b(?:is\s+|was\s+)?allowed\b", r"\b(?:is\s+|was\s+)?(?:forbidden|prohibited|not\s+allowed)\b"),
    (r"\b(?:is\s+|was\s+)?present\b", r"\b(?:is\s+|was\s+)?(?:absent|missing|not\s+present)\b"),
    (r"\b(?:is\s+|was\s+)?mandatory\b", r"\b(?:is\s+|was\s+)?(?:optional|voluntary|not\s+mandatory)\b"),
    (r"\b(?:is\s+|was\s+)?compatible\b", r"\b(?:is\s+|was\s+)?(?:incompatible|not\s+compatible)\b"),
]

NEGATION_WORDS = {"not", "no", "never", "none", "neither", "nor", "cannot", "without", "disapproved", "rejected", "disabled", "unsupported", "impossible", "failure", "failed", "prohibited", "forbidden", "absent", "incompatible"}

SEMANTIC_CONTRADICTION_SYSTEM_PROMPT = """You are a rigorous factual contradiction detector.
Your task is to compare two statements and determine whether they directly CONTRADICT each other on the same claim.

Guidelines:
1. A contradiction exists ONLY if both statements make mutually exclusive, conflicting factual claims (e.g. conflicting measurements, dates, outcomes, definitions, or mutually impossible facts).
2. Do NOT flag complementary or additional information as contradictory.
3. Do NOT flag the same fact expressed in different words, synonyms, or different formats as contradictory.
4. Do NOT flag unrelated information as contradictory.

Return ONLY valid JSON matching this schema:
{
  "contradiction": true|false,
  "reason": "<concise explanation of why the statements conflict or why they do not>",
  "severity": "high"|"medium"|"low"|"none",
  "claim_a": "<specific claim from statement A if conflicting, else empty string>",
  "claim_b": "<specific claim from statement B if conflicting, else empty string>"
}
"""


UNVERIFIED_SOURCE_KEYWORDS = {
    "common belief", "unverified", "rumor", "rumour", "forum", "misconception",
    "popular myth", "incorrect belief", "unverified belief", "community claim",
    "unverified community post", "unverified post"
}


def classify_chunk_authority(chunk: Mapping[str, Any] | Any) -> str:
    """Classifies a chunk as 'authoritative', 'unverified', or 'neutral' based on metadata and text."""
    if hasattr(chunk, "__dict__"):
        d = dict(chunk.__dict__)
    elif isinstance(chunk, Mapping):
        d = dict(chunk)
    else:
        d = {"text": str(chunk)}

    source = str(d.get("source") or d.get("filename") or d.get("title") or "").lower()
    text = str(d.get("text") or "").lower()
    qa = d.get("quality_assessment") or d.get("quality assessment") or {}
    src_quality = str(qa.get("source_quality", "")).lower() if isinstance(qa, Mapping) else ""

    # Check unverified markers in source or text
    for kw in UNVERIFIED_SOURCE_KEYWORDS:
        if kw in source or (kw in text and ("belief is that" in text or "common but incorrect" in text or "unverified" in text or "rumor" in text)):
            return "unverified"
    if src_quality == "low" and ("unverified" in text or "myth" in text or "rumor" in text or "belief" in text):
        return "unverified"

    # Check authoritative markers
    if any(auth_kw in source for auth_kw in ("reference", "knowledge", "official", "textbook", "verified", "authority", "document", "wikipedia", "evidence")):
        return "authoritative"

    return "authoritative" if src_quality in ("high", "medium") else "neutral"


@dataclass(frozen=True)
class ContradictionResult:
    contradiction: bool
    chunk_a: str
    chunk_b: str
    reason: str
    severity: str
    claim_a: str = ""
    claim_b: str = ""
    is_debunked: bool = False
    contradiction_type: str = "fatal"  # "fatal" | "unverified_refuted" | "neutral"
    authoritative_chunk_id: str = ""
    unverified_chunk_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return structured dictionary with standard keys and backwards-compatible aliases."""
        data = asdict(self)
        # Compatibility aliases
        data["chunk_id_a"] = self.chunk_a
        data["chunk_id_b"] = self.chunk_b
        data["explanation"] = self.reason
        data["is_debunked"] = self.is_debunked
        data["contradiction_type"] = self.contradiction_type
        data["authoritative_chunk_id"] = self.authoritative_chunk_id
        data["unverified_chunk_id"] = self.unverified_chunk_id
        return data


def _extract_words(text: str) -> set[str]:
    """Extract lowercase alphanumeric content words, stripping stopwords."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return {w for w in words if w not in STOPWORDS}


def _normalize_numerical_claim(raw_metric: str, raw_value: str) -> tuple[str, str]:
    """Normalize metric name and value (e.g. '85 percent' -> '85%')."""
    metric = re.sub(r"\s+", " ", raw_metric.strip().lower())
    # Strip leading generic noise words from metric
    metric = re.sub(r"^(?:the|an?|reported|measured|achieved|attained|observed)\s+", "", metric)
    value = re.sub(r"\s+", "", raw_value.strip().lower())
    value = value.replace("percent", "%").replace("percentage", "%")
    return metric, value


def _extract_numerical_claims(text: str) -> list[tuple[str, str]]:
    """Extract list of (metric, value) tuples from text (both metric-first and value-first)."""
    claims = []
    # 1. Metric first: "The accuracy is 85%"
    for metric, val in NUMERICAL_CLAIM_PATTERN.findall(text):
        m_norm, v_norm = _normalize_numerical_claim(metric, val)
        if m_norm and v_norm:
            claims.append((m_norm, v_norm))
    # 2. Value first: "achieved 85% accuracy"
    for val, metric in REVERSE_NUMERICAL_CLAIM_PATTERN.findall(text):
        m_norm, v_norm = _normalize_numerical_claim(metric, val)
        if m_norm and v_norm:
            claims.append((m_norm, v_norm))
    return claims


def _have_distinct_modifiers(metric_a: str, metric_b: str) -> bool:
    """Check if metric names belong to distinct modifier subsets (e.g. 'training' vs 'test')."""
    words_a = set(metric_a.split())
    words_b = set(metric_b.split())
    for modifier_group in DISTINCT_MODIFIER_SETS:
        mod_a = words_a.intersection(modifier_group)
        mod_b = words_b.intersection(modifier_group)
        if mod_a and mod_b and mod_a != mod_b:
            return True
    return False


def check_deterministic_contradiction(
    chunk_a_id: str,
    chunk_a_text: str,
    chunk_b_id: str,
    chunk_b_text: str,
) -> ContradictionResult | None:
    """Perform deterministic checks for identity, unrelatedness, or direct numerical conflict.

    Returns a ContradictionResult if a definitive determination can be made without LLM,
    or None if semantic analysis is needed.
    """
    text_a = chunk_a_text.strip()
    text_b = chunk_b_text.strip()

    # 1. Identical or empty content
    if not text_a or not text_b or text_a == text_b or text_a in text_b or text_b in text_a:
        return ContradictionResult(
            contradiction=False,
            chunk_a=chunk_a_id,
            chunk_b=chunk_b_id,
            reason="Chunks contain identical, subsumed, or empty text.",
            severity="none",
        )

    words_a = _extract_words(text_a)
    words_b = _extract_words(text_b)

    # 2. Completely unrelated content (no shared entities/keywords)
    overlap = words_a.intersection(words_b)
    if not overlap:
        return ContradictionResult(
            contradiction=False,
            chunk_a=chunk_a_id,
            chunk_b=chunk_b_id,
            reason="Chunks discuss completely unrelated subjects with no shared entities or claims.",
            severity="none",
        )

    # 3. Direct numerical claim comparison
    claims_a = _extract_numerical_claims(text_a)
    claims_b = _extract_numerical_claims(text_b)

    if claims_a and claims_b:
        for metric_a, val_a in claims_a:
            for metric_b, val_b in claims_b:
                # Do not treat different modifier categories as conflicting (e.g., training vs test)
                if _have_distinct_modifiers(metric_a, metric_b):
                    continue

                words_m_a = set(metric_a.split())
                words_m_b = set(metric_b.split())
                # Match when metric names are identical or have matching root concept
                is_metric_match = (
                    metric_a == metric_b
                    or (words_m_a and words_m_a == words_m_b)
                    or (metric_a in metric_b and len(words_m_a) >= 1)
                    or (metric_b in metric_a and len(words_m_b) >= 1)
                )

                if is_metric_match:
                    if val_a != val_b:
                        # Direct numerical conflict detected
                        return ContradictionResult(
                            contradiction=True,
                            chunk_a=chunk_a_id,
                            chunk_b=chunk_b_id,
                            reason=f"Direct numerical contradiction on '{metric_a}': chunk '{chunk_a_id}' reports '{val_a}', whereas chunk '{chunk_b_id}' reports '{val_b}'.",
                            severity="high",
                            claim_a=f"{metric_a} is {val_a}",
                            claim_b=f"{metric_b} is {val_b}",
                        )
                    else:
                        # Same metric and same value -> Same fact expressed differently/identically
                        return ContradictionResult(
                            contradiction=False,
                            chunk_a=chunk_a_id,
                            chunk_b=chunk_b_id,
                            reason=f"Consistent factual claim: both chunks agree that '{metric_a}' is '{val_a}'.",
                            severity="none",
                            claim_a=f"{metric_a} is {val_a}",
                            claim_b=f"{metric_b} is {val_b}",
                        )

    # 4. Direct polar negation check on similar subjects
    norm_a = re.sub(r"[^\w\s]", "", text_a.lower())
    norm_b = re.sub(r"[^\w\s]", "", text_b.lower())

    for pos_pat, neg_pat in POLAR_NEGATIONS:
        if (re.search(pos_pat, norm_a) and re.search(neg_pat, norm_b)) or (
            re.search(neg_pat, norm_a) and re.search(pos_pat, norm_b)
        ):
            # Check that they share at least 2 content words to ensure it's about the same topic
            if len(overlap) >= 2:
                return ContradictionResult(
                    contradiction=True,
                    chunk_a=chunk_a_id,
                    chunk_b=chunk_b_id,
                    reason="Direct polar negation detected on overlapping topic.",
                    severity="high",
                    claim_a=text_a,
                    claim_b=text_b,
                )

    # 5. Paraphrase detection: identical content words with no negation words in either chunk
    has_negation_a = bool(set(norm_a.split()).intersection(NEGATION_WORDS))
    has_negation_b = bool(set(norm_b.split()).intersection(NEGATION_WORDS))
    if not has_negation_a and not has_negation_b and words_a == words_b and len(words_a) >= 2:
        return ContradictionResult(
            contradiction=False,
            chunk_a=chunk_a_id,
            chunk_b=chunk_b_id,
            reason="Chunks express consistent factual claims with identical key entities.",
            severity="none",
            claim_a=text_a,
            claim_b=text_b,
        )

    return None


class ContradictionDetector:
    """Evaluates evidence chunks for explicit factual contradictions."""

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

    def compare_pair(
        self,
        chunk_a_id: str,
        chunk_a_text: str,
        chunk_b_id: str,
        chunk_b_text: str,
    ) -> ContradictionResult:
        """Compare a single pair of chunks for contradictions."""
        # 1. Try deterministic check first
        deterministic = check_deterministic_contradiction(
            chunk_a_id=chunk_a_id,
            chunk_a_text=chunk_a_text,
            chunk_b_id=chunk_b_id,
            chunk_b_text=chunk_b_text,
        )
        if deterministic is not None:
            return deterministic

        # 2. Semantic LLM analysis via Groq/Llama
        prompt = (
            f"STATEMENT A (chunk_id: {chunk_a_id}):\n{chunk_a_text.strip()}\n\n"
            f"STATEMENT B (chunk_id: {chunk_b_id}):\n{chunk_b_text.strip()}\n\n"
            "Evaluate whether Statement A and Statement B make conflicting, mutually exclusive factual claims."
        )
        messages = [
            {"role": "system", "content": SEMANTIC_CONTRADICTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            if hasattr(self.llm_client, "complete_json") and callable(self.llm_client.complete_json):
                raw_data = self.llm_client.complete_json(messages, model=self.model, max_tokens=400)
            else:
                raw_client = getattr(self.llm_client, "client", self.llm_client)
                response = raw_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=400,
                )
                raw_content = response.choices[0].message.content.strip()
                raw_data = json.loads(raw_content)

            is_contra = bool(raw_data.get("contradiction", False))
            reason = str(raw_data.get("reason", "No contradiction detected."))
            severity = str(raw_data.get("severity", "high" if is_contra else "none")).lower()
            if severity not in ("high", "medium", "low", "none"):
                severity = "high" if is_contra else "none"

            return ContradictionResult(
                contradiction=is_contra,
                chunk_a=chunk_a_id,
                chunk_b=chunk_b_id,
                reason=reason,
                severity=severity,
                claim_a=str(raw_data.get("claim_a", "")),
                claim_b=str(raw_data.get("claim_b", "")),
            )
        except Exception as exc:
            logger.warning(
                "Semantic contradiction LLM analysis failed for %s vs %s: %s",
                chunk_a_id,
                chunk_b_id,
                exc,
            )
            return ContradictionResult(
                contradiction=False,
                chunk_a=chunk_a_id,
                chunk_b=chunk_b_id,
                reason=f"Semantic analysis unavailable ({type(exc).__name__}).",
                severity="none",
            )

    def detect_contradictions(
        self,
        chunks: Sequence[Mapping[str, Any] | Any],
        allow_semantic: bool = True,
    ) -> list[dict[str, Any]]:
        """Compare all relevant retrieved chunks pairwise to detect contradictions."""
        normalized: list[dict[str, Any]] = []
        for idx, item in enumerate(chunks):
            if hasattr(item, "__dict__"):
                d = dict(item.__dict__)
            elif isinstance(item, Mapping):
                d = dict(item)
            else:
                d = {"text": str(item)}

            cid = str(d.get("chunk_id") or f"chunk-{idx:04d}")
            text = str(d.get("text") or "").strip()
            if text:
                d["chunk_id"] = cid
                d["text"] = text
                normalized.append(d)

        from concurrent.futures import ThreadPoolExecutor

        contradictions: list[dict[str, Any]] = []
        semantic_candidates: list[tuple[dict[str, Any], dict[str, Any], float]] = []

        def _enrich_contradiction(
            res: ContradictionResult | dict[str, Any],
            chunk_a_dict: dict[str, Any],
            chunk_b_dict: dict[str, Any],
        ) -> dict[str, Any]:
            c_dict = res.to_dict() if hasattr(res, "to_dict") else dict(res)
            auth_a = classify_chunk_authority(chunk_a_dict)
            auth_b = classify_chunk_authority(chunk_b_dict)
            cid_a = chunk_a_dict["chunk_id"]
            cid_b = chunk_b_dict["chunk_id"]

            if (auth_a == "authoritative" and auth_b == "unverified") or (auth_b == "authoritative" and auth_a == "unverified"):
                auth_id = cid_a if auth_a == "authoritative" else cid_b
                unver_id = cid_b if auth_a == "authoritative" else cid_a
                c_dict["is_debunked"] = True
                c_dict["contradiction_type"] = "unverified_refuted"
                c_dict["severity"] = "low"
                c_dict["authoritative_chunk_id"] = auth_id
                c_dict["unverified_chunk_id"] = unver_id
                logger.info(
                    "Source-aware contradiction: Authoritative chunk '%s' refutes unverified claim in '%s' (non-fatal)",
                    auth_id, unver_id
                )
            else:
                c_dict["is_debunked"] = False
                c_dict["contradiction_type"] = "fatal"
                logger.info(
                    "Fatal contradiction detected between chunk '%s' (auth=%s) and chunk '%s' (auth=%s)",
                    cid_a, auth_a, cid_b, auth_b
                )
            return c_dict

        # Compare each pair once (i < j)
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                chunk_i = normalized[i]
                chunk_j = normalized[j]

                # Fast deterministic check first (regex / polar negations / numerical)
                det = check_deterministic_contradiction(
                    chunk_a_id=chunk_i["chunk_id"],
                    chunk_a_text=chunk_i["text"],
                    chunk_b_id=chunk_j["chunk_id"],
                    chunk_b_text=chunk_j["text"],
                )
                if det is not None:
                    if det.contradiction:
                        contradictions.append(_enrich_contradiction(det, chunk_i, chunk_j))
                    continue

                # Filter pairs with insufficient entity / content overlap
                w_a = _extract_words(chunk_i["text"])
                w_b = _extract_words(chunk_j["text"])
                overlap = w_a.intersection(w_b)
                union = w_a.union(w_b)
                jaccard = len(overlap) / len(union) if union else 0.0

                # Must share at least 2 content words with significant jaccard or >= 4 overlap words
                if (len(overlap) >= 2 and jaccard >= 0.15) or len(overlap) >= 4:
                    semantic_candidates.append((chunk_i, chunk_j, jaccard))

        # Evaluate semantic candidates concurrently, capped at top 2 pairs to prevent LLM latency explosion
        if allow_semantic and semantic_candidates:
            semantic_candidates.sort(key=lambda x: x[2], reverse=True)
            top_candidates = semantic_candidates[:2]

            def _evaluate_candidate(cand: tuple[dict[str, Any], dict[str, Any], float]) -> dict[str, Any] | None:
                ci, cj, _ = cand
                res = self.compare_pair(
                    chunk_a_id=ci["chunk_id"],
                    chunk_a_text=ci["text"],
                    chunk_b_id=cj["chunk_id"],
                    chunk_b_text=cj["text"],
                )
                if res.contradiction:
                    return _enrich_contradiction(res, ci, cj)
                return None

            with ThreadPoolExecutor(max_workers=min(2, len(top_candidates))) as executor:
                for enriched in executor.map(_evaluate_candidate, top_candidates):
                    if enriched is not None:
                        contradictions.append(enriched)

        return contradictions


_contradiction_detector: ContradictionDetector | None = None


def get_contradiction_detector() -> ContradictionDetector:
    """Return process-wide singleton ContradictionDetector instance."""
    global _contradiction_detector
    if _contradiction_detector is None:
        _contradiction_detector = ContradictionDetector()
    return _contradiction_detector
