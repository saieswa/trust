"use client";

export default function AcademicScorecard({ trustScore, decision, verificationStatus, revisionCount }) {
  // Extract numerical score safely
  let scoreVal = null;
  let scoringMethod = "Calibrated Assessment";
  let thresholdLabel = "EVALUATING";
  let explanation = "";

  if (trustScore != null) {
    if (typeof trustScore === "number") {
      scoreVal = trustScore;
    } else if (typeof trustScore === "object") {
      scoreVal = trustScore.overall_score != null ? trustScore.overall_score : null;
      scoringMethod = trustScore.scoring_method || scoringMethod;
      thresholdLabel = trustScore.threshold_label || thresholdLabel;
      explanation = trustScore.explanation || "";
    }
  }

  // Derive level if not provided
  if (thresholdLabel === "EVALUATING" && scoreVal != null) {
    if (scoreVal >= 0.75) thresholdLabel = "HIGH";
    else if (scoreVal >= 0.50) thresholdLabel = "MEDIUM";
    else thresholdLabel = "LOW";
  }

  const scorePct = scoreVal != null ? Math.round(scoreVal * 100) : null;
  const levelClass = thresholdLabel.toLowerCase();

  // Verification status details
  const vStatus = (verificationStatus || "PENDING").toUpperCase();
  let vStatusClass = "neutral";
  let vStatusLabel = vStatus;

  if (vStatus === "SUPPORTED") {
    vStatusClass = "supported";
    vStatusLabel = "Verified Supported";
  } else if (vStatus === "PARTIALLY_SUPPORTED") {
    vStatusClass = "partial";
    vStatusLabel = "Partially Supported";
  } else if (vStatus === "UNSUPPORTED" || vStatus === "FAILED") {
    vStatusClass = "unsupported";
    vStatusLabel = vStatus === "FAILED" ? "Verification Failed" : "Unsupported Claims";
  } else if (vStatus === "ABSTAINED") {
    vStatusClass = "abstained";
    vStatusLabel = "Abstained (Safety Guard)";
  }

  // Method formatting
  const formattedMethod = scoringMethod.replace(/_/g, " ").toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase());

  return (
    <div className="academic-scorecard">
      <div className="scorecard-metric">
        <span className="metric-eyebrow">Trust Assessment</span>
        <div className="metric-main">
          <div className="score-value">
            {scorePct != null ? `${scorePct}%` : "—"}
          </div>
          <div className="score-bar-wrapper">
            <div
              className={`score-bar-fill level-${levelClass}`}
              style={{ width: `${Math.max(0, Math.min(100, scorePct ?? 0))}%` }}
            />
          </div>
        </div>
        <span className="metric-caption">{formattedMethod}</span>
      </div>

      <div className="scorecard-metric">
        <span className="metric-eyebrow">Trust Level</span>
        <div className="metric-main">
          <span className={`trust-level-pill level-${levelClass}`}>
            <span className="level-dot" />
            {thresholdLabel}
          </span>
        </div>
        <span className="metric-caption">
          {thresholdLabel === "HIGH" && "Score ≥ 75% • Proceeded to Synthesis"}
          {thresholdLabel === "MEDIUM" && "50% ≤ Score < 75% • Retrieval Expansion"}
          {thresholdLabel === "LOW" && "Score < 50% • Hallucination Guard Active"}
        </span>
      </div>

      <div className="scorecard-metric">
        <span className="metric-eyebrow">Verification Status</span>
        <div className="metric-main">
          <span className={`verification-badge vstatus-${vStatusClass}`}>
            {vStatusClass === "supported" && "✓ "}
            {vStatusClass === "unsupported" && "✕ "}
            {vStatusClass === "partial" && "⚠ "}
            {vStatusLabel}
          </span>
        </div>
        <span className="metric-caption">
          {revisionCount != null && revisionCount > 0
            ? `${revisionCount} controlled revision performed`
            : "Atomic claim verification against evidence"}
        </span>
      </div>
    </div>
  );
}
