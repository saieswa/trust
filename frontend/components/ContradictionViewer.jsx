"use client";

export default function ContradictionViewer({ contradictions, retrievalResults = [] }) {
  if (!contradictions || contradictions.length === 0) {
    return null;
  }

  // Helper to find chunk details from retrievalResults
  const findChunk = (chunkId) => {
    if (!chunkId) return null;
    const cid = String(chunkId);
    return retrievalResults.find((r) => String(r.chunk_id) === cid);
  };

  return (
    <section className="contradiction-alert-section" aria-label="Contradiction Warning">
      <div className="contradiction-alert-header">
        <span className="warning-symbol" aria-hidden="true">⚠</span>
        <div className="warning-title-block">
          <h4>Contradictory Evidence</h4>
          <p className="warning-subtitle">
            The system detected {contradictions.length} factual conflict{contradictions.length === 1 ? "" : "s"} among the retrieved evidence passages. Conflicting evidence is preserved below for academic transparency.
          </p>
        </div>
      </div>

      <div className="contradiction-items-list">
        {contradictions.map((c, index) => {
          const chunkAId = c.chunk_a || c.chunk_id_a;
          const chunkBId = c.chunk_b || c.chunk_id_b;
          const chunkA = findChunk(chunkAId);
          const chunkB = findChunk(chunkBId);

          const sourceAName = chunkA?.filename || chunkA?.source || c.source_a || `Document Chunk ${chunkAId || "A"}`;
          const sourceAPage = chunkA?.page_number != null ? `Page ${chunkA.page_number}` : (c.page_a != null ? `Page ${c.page_a}` : null);
          const textA = c.claim_a || chunkA?.text || c.text_a || "No passage text available.";

          const sourceBName = chunkB?.filename || chunkB?.source || c.source_b || `Document Chunk ${chunkBId || "B"}`;
          const sourceBPage = chunkB?.page_number != null ? `Page ${chunkB.page_number}` : (c.page_b != null ? `Page ${c.page_b}` : null);
          const textB = c.claim_b || chunkB?.text || c.text_b || "No passage text available.";

          const reason = c.reason || c.explanation || "Conflicting factual claims detected on the same topic.";
          const severity = (c.severity || "high").toUpperCase();

          return (
            <div className="contradiction-comparison-card" key={`contra-${index}`}>
              <div className="comparison-meta-row">
                <span className="conflict-tag">Conflict #{index + 1}</span>
                <span className="conflict-severity">Severity: {severity}</span>
              </div>

              <div className="conflict-reason-box">
                <strong>Disagreement:</strong> {reason}
              </div>

              <div className="comparison-sources-grid">
                {/* SOURCE A */}
                <div className="source-box source-a">
                  <div className="source-header">
                    <span className="source-label">Source A:</span>
                    <span className="source-citation">
                      {sourceAName} {sourceAPage ? `(${sourceAPage})` : ""}
                    </span>
                  </div>
                  <blockquote className="source-quote">
                    "{textA}"
                  </blockquote>
                </div>

                {/* VS DIVIDER */}
                <div className="vs-divider" aria-hidden="true">
                  <span>VS</span>
                </div>

                {/* SOURCE B */}
                <div className="source-box source-b">
                  <div className="source-header">
                    <span className="source-label">Source B:</span>
                    <span className="source-citation">
                      {sourceBName} {sourceBPage ? `(${sourceBPage})` : ""}
                    </span>
                  </div>
                  <blockquote className="source-quote">
                    "{textB}"
                  </blockquote>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
