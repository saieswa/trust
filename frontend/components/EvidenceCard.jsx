"use client";

import { useState } from "react";

export default function EvidenceCard({
  item,
  evaluation,
  contradiction,
  defaultExpanded = true,
}) {
  const [expanded, setExpanded] = useState(defaultExpanded);

  // 1. Source
  const sourceName = item.filename || item.source || item.document_id || "Uploaded Document";

  // 2. Page Number
  const pageLabel = item.page_number != null ? `Page ${item.page_number}` : "Page —";

  // 3. Relevance Score
  let relevanceVal = null;
  if (evaluation?.relevance_score != null) {
    relevanceVal = Math.round(evaluation.relevance_score * 100);
  } else if (item.relevance_score != null) {
    relevanceVal = Math.round(item.relevance_score * 100);
  } else if (item.score != null) {
    relevanceVal = Math.round(item.score * 100);
  }

  // 4. Support Status
  const rawSupport = evaluation?.support_status || (evaluation?.supports_question ? "supports" : "neutral");
  let supportLabel = "Reviewed";
  let supportType = "neutral";

  if (rawSupport === "supports" || evaluation?.supports_question === true) {
    supportLabel = "Supports Question";
    supportType = "supported";
  } else if (rawSupport === "conflicts" || item.has_contradiction) {
    supportLabel = "Conflicting Evidence";
    supportType = "conflict";
  } else if (rawSupport === "unrelated_document" || rawSupport === "rejected" || evaluation?.relevance === false) {
    supportLabel = "Rejected / Unrelated";
    supportType = "rejected";
  } else {
    supportLabel = "Neutral Evidence";
    supportType = "neutral";
  }

  // 5. Evidence text
  const evidenceText = item.text || "No text available for this chunk.";
  const previewText = evidenceText.length > 200 ? `${evidenceText.slice(0, 200)}...` : evidenceText;

  // Contradiction presence
  const isContradicted = Boolean(item.has_contradiction || contradiction);

  return (
    <article
      className={`academic-evidence-card ${isContradicted ? "card-has-conflict" : ""} ${expanded ? "is-expanded" : "is-collapsed"}`}
      id={`evidence-${item.chunk_id}`}
    >
      {/* CARD HEADER / SUMMARY ROW */}
      <header
        className="evidence-card-header"
        onClick={() => setExpanded((prev) => !prev)}
        role="button"
        tabIndex={0}
        aria-expanded={expanded}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            setExpanded((prev) => !prev);
          }
        }}
      >
        <div className="card-header-left">
          <span className="source-badge" title={sourceName}>
            <span className="doc-icon" aria-hidden="true">📄</span>
            <strong className="source-title">{sourceName}</strong>
          </span>
          <span className="page-badge">{pageLabel}</span>
        </div>

        <div className="card-header-right">
          {relevanceVal != null && (
            <span className="relevance-pill" title="Vector & Critic Relevance Score">
              {relevanceVal}% Relevance
            </span>
          )}

          <span className={`support-pill support-${supportType}`}>
            {supportType === "supported" && "✓ "}
            {supportType === "conflict" && "⚠ "}
            {supportType === "rejected" && "✕ "}
            {supportLabel}
          </span>

          <button
            type="button"
            className="expand-toggle-btn"
            aria-label={expanded ? "Collapse evidence passage" : "Expand evidence passage"}
            onClick={(e) => {
              e.stopPropagation();
              setExpanded((prev) => !prev);
            }}
          >
            <span className="toggle-text">{expanded ? "Collapse" : "Expand"}</span>
            <span className="chevron-icon" aria-hidden="true">{expanded ? "▲" : "▼"}</span>
          </button>
        </div>
      </header>

      {/* EXPANDABLE BODY */}
      <div className="evidence-card-body">
        {/* If collapsed, show short excerpt preview */}
        {!expanded && (
          <p className="collapsed-preview" onClick={() => setExpanded(true)}>
            "{previewText}" <span className="read-more-hint">[Click to read full passage]</span>
          </p>
        )}

        {/* If expanded, show full text, evaluation notes, and any conflict details */}
        {expanded && (
          <div className="expanded-content">
            <div className="evidence-text-block">
              <span className="quote-mark">“</span>
              <p className="full-evidence-text">{evidenceText}</p>
              <span className="quote-mark end-quote">”</span>
            </div>

            {/* Critic Evaluation Explanation if present */}
            {evaluation?.explanation && (
              <div className="critic-evaluation-note">
                <span className="critic-label">Critic Assessment:</span>
                <p className="critic-explanation">{evaluation.explanation}</p>
              </div>
            )}

            {/* EMBEDDED CONTRADICTION WARNING IF PRESENT ON THIS CHUNK */}
            {isContradicted && (
              <div className="card-contradiction-box">
                <div className="card-contradiction-header">
                  <span className="card-contra-icon">⚠</span>
                  <strong>Contradictory Evidence</strong>
                </div>

                {item.contradiction_reason && (
                  <p className="card-contra-reason">
                    <strong>Conflict:</strong> {item.contradiction_reason}
                  </p>
                )}

                {contradiction ? (
                  <div className="contra-source-breakdown">
                    <div className="contra-subsource">
                      <span className="contra-source-tag">Source A:</span>
                      <p className="contra-source-text">
                        "{contradiction.claim_a || contradiction.text_a || "Claim A"}"
                      </p>
                    </div>
                    <div className="contra-subsource">
                      <span className="contra-source-tag">Source B:</span>
                      <p className="contra-source-text">
                        "{contradiction.claim_b || contradiction.text_b || "Claim B"}"
                      </p>
                    </div>
                  </div>
                ) : (
                  <div className="contra-source-breakdown">
                    <div className="contra-subsource">
                      <span className="contra-source-tag">Source A:</span>
                      <p className="contra-source-text">
                        {sourceName} ({pageLabel}): "{evidenceText.slice(0, 150)}..."
                      </p>
                    </div>
                    <div className="contra-subsource">
                      <span className="contra-source-tag">Source B:</span>
                      <p className="contra-source-text">
                        Conflicting document passage detected during multi-chunk cross-evaluation.
                      </p>
                    </div>
                  </div>
                )}
              </div>
            )}

            <footer className="evidence-card-footer">
              <span className="chunk-id-tag">ID: {item.chunk_id}</span>
              {evaluation?.quality_assessment && (
                <span className="quality-tag">
                  Quality: {String(evaluation.quality_assessment).toUpperCase()}
                </span>
              )}
            </footer>
          </div>
        )}
      </div>
    </article>
  );
}
