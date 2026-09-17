"""Trust Score Model and Explanation for Trust-Aware RAG.

Calculates an objective, calibrated Trust Score T in [0.0, 1.0] derived
from explicit, measurable evidence features:
1. Retrieval relevance (f_rel)
2. Evidence support (f_supp)
3. Source quality & evidence strength (f_qual)
4. Evidence agreement / consensus (f_agree)
5. Contradiction ratio (f_contra)
6. Retrieval confidence (f_conf)
7. Verifier agreement (f_ver) when available

Architecture:
- The score is strictly computed from measurable features (never subjective LLM prompting).
- Supports machine-learned prediction via trained XGBoost regressor when present.
- If XGBoost is not trained or unavailable, transparently falls back to the calibrated weighted formula.
- ML prediction and fallback scoring are kept cleanly separated.
- Configurable operational thresholds (0.75 for HIGH_TRUST, 0.50 for MODERATE/LOW_TRUST)
  are explicitly labeled as configurable heuristics rather than scientifically validated constants.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

# Configurable operational thresholds
# NOTE: The default 0.75 and 0.50 thresholds are heuristic operational boundaries.
# They are NOT empirically or scientifically calibrated until dedicated calibration experiments are completed.
DEFAULT_CONFIGURABLE_HIGH_THRESHOLD: float = 0.75
DEFAULT_CONFIGURABLE_LOW_THRESHOLD: float = 0.50

HIGH_TRUST_THRESHOLD: float = DEFAULT_CONFIGURABLE_HIGH_THRESHOLD
MODERATE_TRUST_THRESHOLD: float = DEFAULT_CONFIGURABLE_LOW_THRESHOLD

STRENGTH_WEIGHTS = {
    "high": 1.0,
    "medium": 0.70,
    "low": 0.35,
    "none": 0.0,
}

QUALITY_WEIGHTS = {
    "high": 1.0,
    "medium": 0.75,
    "low": 0.40,
}


@dataclass(frozen=True)
class TrustFeatures:
    """Explicit measurable features used to compute the Trust Score."""
    relevance: float  # [0.0, 1.0]
    evidence_support: float  # [0.0, 1.0]
    source_quality: float  # [0.0, 1.0]
    evidence_agreement: float  # [0.0, 1.0]
    contradiction_ratio: float  # [0.0, 1.0]
    retrieval_confidence: float  # [0.0, 1.0]
    verifier_agreement: float | None = None  # [0.0, 1.0] or None if unverified

    def to_dict(self) -> dict[str, Any]:
        return {
            "relevance": round(self.relevance, 3),
            "evidence_support": round(self.evidence_support, 3),
            "source_quality": round(self.source_quality, 3),
            "evidence_agreement": round(self.evidence_agreement, 3),
            "contradiction_ratio": round(self.contradiction_ratio, 3),
            "retrieval_confidence": round(self.retrieval_confidence, 3),
            "verifier_agreement": (
                round(self.verifier_agreement, 3)
                if self.verifier_agreement is not None
                else None
            ),
        }


@dataclass(frozen=True)
class TrustComponents:
    """Component pillar scores preserved for backwards compatibility."""
    relevance_score: float
    source_quality_score: float
    consistency_score: float
    freshness_score: float


@dataclass(frozen=True)
class TrustFactors:
    """Fact counts for evidence chunk breakdown."""
    total_chunks: int
    relevant_chunks: int
    supporting_chunks: int
    contradicting_chunks: int
    outdated_chunks: int
    unrelated_chunks: int


@dataclass(frozen=True)
class TrustScoreResult:
    """Complete Trust Score result with features, method, and explanation."""
    trust_score: float
    overall_score: float  # Compatibility alias for trust_score
    trust_percentage: int
    trust_level: str  # HIGH_TRUST, MODERATE_TRUST, LOW_TRUST
    features: TrustFeatures
    scoring_method: str  # 'ml_xgboost' or 'weighted_fallback'
    explanation: str
    components: TrustComponents
    factors: TrustFactors
    rationale: str  # Compatibility alias for explanation
    warnings: list[str]
    high_threshold: float = DEFAULT_CONFIGURABLE_HIGH_THRESHOLD
    low_threshold: float = DEFAULT_CONFIGURABLE_LOW_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        """Return structured dictionary matching user specification and backwards compatibility."""
        return {
            # Top-level requested fields
            "trust_score": self.trust_score,
            "overall_score": self.overall_score,
            "relevance": self.features.relevance,
            "evidence_support": self.features.evidence_support,
            "source_quality": self.features.source_quality,
            "evidence_agreement": self.features.evidence_agreement,
            "contradiction_ratio": self.features.contradiction_ratio,
            "retrieval_confidence": self.features.retrieval_confidence,
            "verifier_agreement": self.features.verifier_agreement,
            "scoring_method": self.scoring_method,
            "explanation": self.explanation,
            # Pipeline & decision compatibility
            "trust_level": self.trust_level,
            "trust_percentage": self.trust_percentage,
            "features": self.features.to_dict(),
            "components": asdict(self.components),
            "factors": asdict(self.factors),
            "rationale": self.rationale,
            "warnings": list(self.warnings),
            "threshold_config": {
                "high_threshold": self.high_threshold,
                "low_threshold": self.low_threshold,
                "calibration_status": "configurable_heuristic_uncalibrated",
                "notes": (
                    "Thresholds are operational defaults (high=0.75, low=0.50) "
                    "pending empirical dataset calibration experiments."
                ),
            },
        }


from functools import lru_cache


@lru_cache(maxsize=4)
def _get_cached_booster(model_path_str: str) -> Any:
    """Load and cache XGBoost booster in memory to avoid repeated disk reads."""
    import xgboost as xgb  # type: ignore
    booster = xgb.Booster()
    booster.load_model(model_path_str)
    return booster


# ==============================================================================
# 1. Machine Learning Prediction (XGBoost)
# ==============================================================================
def predict_xgboost_trust_score(
    features: TrustFeatures,
    model_path: str | Path | None = None,
) -> float | None:
    """Predict Trust Score using a trained XGBoost regressor model.

    Kept strictly separated from fallback calculation.
    Returns float in [0.0, 1.0] if model exists and prediction succeeds.
    Returns None if XGBoost is uninstalled, untrained, or fails prediction.
    """
    try:
        import xgboost as xgb  # type: ignore
    except ImportError:
        logger.debug("XGBoost library not installed; using fallback scoring formula.")
        return None

    resolved_path = model_path
    if resolved_path is None:
        default_path = Path(__file__).parent.parent / "models" / "trust_xgboost.json"
        if default_path.exists():
            resolved_path = default_path

    if not resolved_path or not Path(resolved_path).exists():
        logger.debug("XGBoost model file not found at %s; using fallback scoring.", resolved_path)
        return None

    try:
        booster = _get_cached_booster(str(resolved_path))

        vector = [
            features.relevance,
            features.evidence_support,
            features.source_quality,
            features.evidence_agreement,
            features.contradiction_ratio,
            features.retrieval_confidence,
            features.verifier_agreement if features.verifier_agreement is not None else 0.50,
        ]
        import numpy as np
        dmatrix = xgb.DMatrix(np.array([vector]))
        pred = booster.predict(dmatrix)
        val = float(pred[0])
        if 0.0 <= val <= 1.0:
            return round(val, 3)
        logger.warning("XGBoost prediction %f out of [0, 1] bounds; using fallback.", val)
        return None
    except Exception as exc:
        logger.warning("XGBoost inference failed: %s; using fallback scoring.", exc)
        return None


# ==============================================================================
# 2. Explicit Weighted Fallback Formula
# ==============================================================================
def calculate_fallback_trust_score(
    features: TrustFeatures,
    outdated_ratio: float = 0.0,
) -> float:
    """Explicit, calibrated mathematical fallback formula for Trust Score calculation.

    Combines measurable signals without subjective LLM prompting:
    - Base quality: relevance (35%), evidence support (25%), source quality (25%), retrieval confidence (15%).
      If verifier agreement is available, it is incorporated with 20% weight.
    - Contradiction multiplier: severe penalty for conflicting claims.
    - Evidence agreement multiplier: reward for consensus across evidence passages.
    - Temporal freshness multiplier: discount for outdated information.
    """
    # 1. Base Quality
    base_quality = (
        0.35 * features.relevance
        + 0.25 * features.evidence_support
        + 0.25 * features.source_quality
        + 0.15 * features.retrieval_confidence
    )

    if features.verifier_agreement is not None:
        base_quality = 0.80 * base_quality + 0.20 * features.verifier_agreement

    # 2. Contradiction Penalty Multiplier: severe penalty for conflicting claims
    # If contradiction_ratio is 0 -> multiplier 1.0; if 0.5 -> multiplier 0.575; if 1.0 -> multiplier 0.15
    contradiction_multiplier = max(0.05, 1.0 - 0.85 * features.contradiction_ratio)

    # 3. Evidence Agreement Multiplier: consensus across passages
    agreement_multiplier = 0.60 + 0.40 * features.evidence_agreement

    # 4. Outdatedness Penalty Multiplier
    freshness_multiplier = max(0.20, 1.0 - 0.35 * min(1.0, max(0.0, outdated_ratio)))

    # Final product, bounded strictly in [0.0, 1.0]
    raw_trust = base_quality * contradiction_multiplier * agreement_multiplier * freshness_multiplier
    return round(min(1.0, max(0.0, raw_trust)), 3)


# ==============================================================================
# 3. Trust Score Calculation Orchestrator
# ==============================================================================
def calculate_trust_score(
    evaluations: Sequence[Mapping[str, Any]],
    contradictions: Sequence[Mapping[str, Any]] | None = None,
    retrieval_scores: Sequence[float] | None = None,
    verifier_score: float | None = None,
    high_threshold: float = HIGH_TRUST_THRESHOLD,
    low_threshold: float = MODERATE_TRUST_THRESHOLD,
    ml_model_path: str | Path | None = None,
) -> TrustScoreResult:
    """Calculate the calibrated Trust Score from explicit measurable features.

    Uses XGBoost model if trained and available; otherwise uses explicit weighted fallback formula.
    """
    contradictions = list(contradictions or [])
    total_chunks = len(evaluations)

    # Base zero-chunk case
    if total_chunks == 0:
        empty_features = TrustFeatures(
            relevance=0.0,
            evidence_support=0.0,
            source_quality=0.0,
            evidence_agreement=0.0,
            contradiction_ratio=0.0,
            retrieval_confidence=0.0,
            verifier_agreement=None,
        )
        return TrustScoreResult(
            trust_score=0.0,
            overall_score=0.0,
            trust_percentage=0,
            trust_level="LOW_TRUST",
            features=empty_features,
            scoring_method="weighted_fallback",
            explanation="No evidence chunks were available to evaluate. Zero trust assigned.",
            components=TrustComponents(
                relevance_score=0.0,
                source_quality_score=0.0,
                consistency_score=1.0,
                freshness_score=1.0,
            ),
            factors=TrustFactors(
                total_chunks=0,
                relevant_chunks=0,
                supporting_chunks=0,
                contradicting_chunks=0,
                outdated_chunks=0,
                unrelated_chunks=0,
            ),
            rationale="No evidence chunks were available to evaluate. Zero trust assigned.",
            warnings=["No evidence available."],
            high_threshold=high_threshold,
            low_threshold=low_threshold,
        )

    # Filter out unrelated cross-document chunks
    valid_chunks = [
        e for e in evaluations
        if e.get("support_status") != "unrelated_document"
        and e.get("support status") != "unrelated_document"
    ]
    unrelated_count = total_chunks - len(valid_chunks)

    if len(valid_chunks) == 0:
        empty_features = TrustFeatures(
            relevance=0.0,
            evidence_support=0.0,
            source_quality=0.0,
            evidence_agreement=0.0,
            contradiction_ratio=0.0,
            retrieval_confidence=0.0,
            verifier_agreement=None,
        )
        return TrustScoreResult(
            trust_score=0.0,
            overall_score=0.0,
            trust_percentage=0,
            trust_level="LOW_TRUST",
            features=empty_features,
            scoring_method="weighted_fallback",
            explanation="All retrieved evidence belongs to unrelated documents. Cross-document evidence strictly rejected.",
            components=TrustComponents(
                relevance_score=0.0,
                source_quality_score=0.0,
                consistency_score=0.0,
                freshness_score=1.0,
            ),
            factors=TrustFactors(
                total_chunks=total_chunks,
                relevant_chunks=0,
                supporting_chunks=0,
                contradicting_chunks=0,
                outdated_chunks=0,
                unrelated_chunks=unrelated_count,
            ),
            rationale="All retrieved evidence belongs to unrelated documents. Cross-document evidence strictly rejected.",
            warnings=["All evidence chunks belong to unrelated documents."],
            high_threshold=high_threshold,
            low_threshold=low_threshold,
        )

    # 1. Feature: Retrieval Relevance (f_rel)
    relevant_chunks = [e for e in valid_chunks if e.get("relevance", False)]
    relevance_ratio = len(relevant_chunks) / len(valid_chunks)

    if retrieval_scores and len(retrieval_scores) == len(evaluations):
        valid_retrieval_scores = [
            score for e, score in zip(evaluations, retrieval_scores)
            if e in valid_chunks and e.get("relevance", False)
        ]
        if valid_retrieval_scores:
            avg_similarity = sum(valid_retrieval_scores) / len(valid_retrieval_scores)
            f_rel = 0.5 * relevance_ratio + 0.5 * min(1.0, max(0.0, avg_similarity))
        else:
            f_rel = 0.5 * relevance_ratio
    else:
        f_rel = relevance_ratio

    # 2. Feature: Evidence Support (f_supp)
    supporting_chunks = [
        e for e in valid_chunks
        if str(e.get("support_status", "")).lower() == "supported"
    ]
    f_supp = len(supporting_chunks) / len(valid_chunks)

    # 3. Feature: Source Quality & Evidence Strength (f_qual)
    eval_target = relevant_chunks if relevant_chunks else valid_chunks
    quality_scores: list[float] = []
    for item in eval_target:
        qa = item.get("quality_assessment") or item.get("quality assessment") or {}
        if not isinstance(qa, Mapping):
            qa = {}
        strength = str(qa.get("evidence_strength", "medium")).lower()
        src_qual = str(qa.get("source_quality", "medium")).lower()
        w_strength = STRENGTH_WEIGHTS.get(strength, 0.50)
        w_source = QUALITY_WEIGHTS.get(src_qual, 0.60)
        quality_scores.append(0.55 * w_strength + 0.45 * w_source)

    f_qual = sum(quality_scores) / len(quality_scores) if quality_scores else 0.50

    # 4. Feature: Contradiction Ratio (f_contra)
    contradiction_count = len(contradictions)
    chunk_contra_count = sum(
        1 for e in valid_chunks
        if "contradiction" in str(e.get("contradiction_status", "")).lower()
        or str(e.get("support_status", "")).lower() == "contradicted"
    )
    total_contradictions = max(contradiction_count, chunk_contra_count)
    f_contra = min(1.0, total_contradictions / len(valid_chunks))

    # 5. Feature: Evidence Agreement (f_agree)
    if total_contradictions > 0:
        f_agree = max(0.0, 1.0 - 1.2 * f_contra)
    else:
        f_agree = min(1.0, 0.70 + 0.30 * f_supp) if f_rel > 0 else 0.0

    # 6. Feature: Retrieval Confidence (f_conf)
    if retrieval_scores and len(retrieval_scores) > 0:
        f_conf = min(1.0, max(0.0, sum(retrieval_scores) / len(retrieval_scores)))
    else:
        f_conf = 0.80

    # 7. Feature: Verifier Agreement (f_ver)
    f_ver = min(1.0, max(0.0, verifier_score)) if verifier_score is not None else None

    # Outdated chunks analysis
    outdated_count = sum(
        1 for e in valid_chunks
        if isinstance(e.get("quality_assessment") or e.get("quality assessment"), Mapping)
        and (e.get("quality_assessment") or e.get("quality assessment")).get("potential_outdated", False)
    )
    outdated_ratio = outdated_count / len(valid_chunks)

    # Legacy component metrics
    s_cons = max(0.05, 1.0 - 0.45 * total_contradictions) if total_contradictions > 0 else 1.0
    s_fresh = max(0.20, 1.0 - 0.35 * outdated_ratio) if outdated_count > 0 else 1.0

    features = TrustFeatures(
        relevance=f_rel,
        evidence_support=f_supp,
        source_quality=f_qual,
        evidence_agreement=f_agree,
        contradiction_ratio=f_contra,
        retrieval_confidence=f_conf,
        verifier_agreement=f_ver,
    )

    # Score prediction: ML first, fallback second
    ml_score = predict_xgboost_trust_score(features, model_path=ml_model_path)
    if ml_score is not None:
        trust_score = ml_score
        scoring_method = "ml_xgboost"
    else:
        trust_score = calculate_fallback_trust_score(features, outdated_ratio=outdated_ratio)
        scoring_method = "weighted_fallback"

    overall_score = trust_score
    trust_percentage = int(round(trust_score * 100))

    # Categorization based on configurable thresholds
    if overall_score >= high_threshold and total_contradictions == 0:
        trust_level = "HIGH_TRUST"
    elif overall_score >= low_threshold:
        trust_level = "MODERATE_TRUST"
    else:
        trust_level = "LOW_TRUST"

    warnings: list[str] = []
    if total_contradictions > 0:
        warnings.append(
            f"Cross-chunk factual contradiction detected ({total_contradictions} instance(s)). Consistency penalized."
        )
    if outdated_count > 0:
        warnings.append(
            f"{outdated_count} chunk(s) flagged as potentially containing outdated information."
        )
    if unrelated_count > 0:
        warnings.append(
            f"{unrelated_count} chunk(s) belonged to unrelated documents and were strictly excluded."
        )
    if len(relevant_chunks) == 0:
        warnings.append("No relevant chunks found in the retrieved evidence.")

    explanation = _generate_rationale(
        overall_score=overall_score,
        trust_level=trust_level,
        s_rel=f_rel,
        s_qual=f_qual,
        s_cons=s_cons,
        s_fresh=s_fresh,
        relevant_count=len(relevant_chunks),
        total_valid=len(valid_chunks),
        total_contradictions=total_contradictions,
        outdated_count=outdated_count,
        scoring_method=scoring_method,
    )

    return TrustScoreResult(
        trust_score=trust_score,
        overall_score=overall_score,
        trust_percentage=trust_percentage,
        trust_level=trust_level,
        features=features,
        scoring_method=scoring_method,
        explanation=explanation,
        components=TrustComponents(
            relevance_score=round(f_rel, 3),
            source_quality_score=round(f_qual, 3),
            consistency_score=round(s_cons, 3),
            freshness_score=round(s_fresh, 3),
        ),
        factors=TrustFactors(
            total_chunks=total_chunks,
            relevant_chunks=len(relevant_chunks),
            supporting_chunks=len(supporting_chunks),
            contradicting_chunks=total_contradictions,
            outdated_chunks=outdated_count,
            unrelated_chunks=unrelated_count,
        ),
        rationale=explanation,
        warnings=warnings,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
    )


def explain_trust_score(result: TrustScoreResult) -> dict[str, Any]:
    """Return a detailed human-interpretable explanation of the Trust Score."""
    return result.to_dict()


def _generate_rationale(
    overall_score: float,
    trust_level: str,
    s_rel: float,
    s_qual: float,
    s_cons: float,
    s_fresh: float,
    relevant_count: int,
    total_valid: int,
    total_contradictions: int,
    outdated_count: int,
    scoring_method: str = "weighted_fallback",
) -> str:
    """Compose concise, clear rationale explaining the Trust Score calculation."""
    method_str = "ML (XGBoost)" if scoring_method == "ml_xgboost" else "weighted formula"

    if total_contradictions > 0:
        return (
            f"Assigned {trust_level} ({overall_score:.1%}, via {method_str}): Factual contradictions detected "
            f"among retrieved evidence (consistency score {s_cons:.2f}). Evidence is conflicting."
        )
    if relevant_count == 0:
        return (
            f"Assigned {trust_level} ({overall_score:.1%}, via {method_str}): None of the {total_valid} valid chunks "
            f"directly answer or support the question (relevance score 0.00)."
        )
    if trust_level == "HIGH_TRUST":
        return (
            f"Assigned HIGH_TRUST ({overall_score:.1%}, via {method_str}): {relevant_count}/{total_valid} chunks are relevant "
            f"(relevance {s_rel:.2f}), strong source quality ({s_qual:.2f}), with zero contradictions."
        )
    if trust_level == "MODERATE_TRUST":
        notes = []
        if s_rel < 0.60:
            notes.append(f"partial relevance ({s_rel:.2f})")
        if s_qual < 0.60:
            notes.append(f"moderate quality ({s_qual:.2f})")
        if outdated_count > 0:
            notes.append(f"{outdated_count} outdated chunk(s)")
        note_text = ", ".join(notes) if notes else "evidence is borderline"
        return (
            f"Assigned MODERATE_TRUST ({overall_score:.1%}, via {method_str}): {note_text}. "
            "Additional evidence recommended."
        )

    return (
        f"Assigned LOW_TRUST ({overall_score:.1%}, via {method_str}): Insufficient evidence strength or relevance "
        "to reliably answer without hallucination."
    )
