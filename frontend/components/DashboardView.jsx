"use client";

import { useEffect, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export default function DashboardView({ activeQueryInfo }) {
  const [stats, setStats] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [selectedDocFilter, setSelectedDocFilter] = useState("all");
  const [dateFilter, setDateFilter] = useState("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  // Inspect a specific historical query or the active chat query
  const [inspectedQuery, setInspectedQuery] = useState(activeQueryInfo || null);

  useEffect(() => {
    if (activeQueryInfo) {
      setInspectedQuery(activeQueryInfo);
    }
  }, [activeQueryInfo]);

  const fetchStats = async () => {
    setLoading(true);
    setError("");
    try {
      const params = new URLSearchParams();
      if (selectedDocFilter && selectedDocFilter !== "all") {
        params.append("document_id", selectedDocFilter);
      }

      const now = new Date();
      if (dateFilter === "today") {
        const todayStr = new Date(now.getFullYear(), now.getMonth(), now.getDate()).toISOString();
        params.append("start_date", todayStr);
      } else if (dateFilter === "7days") {
        const sevenDaysAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000).toISOString();
        params.append("start_date", sevenDaysAgo);
      } else if (dateFilter === "30days") {
        const thirtyDaysAgo = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000).toISOString();
        params.append("start_date", thirtyDaysAgo);
      }

      const res = await fetch(`${API_URL}/api/dashboard/stats?${params.toString()}`);
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      const data = await res.json();
      setStats(data);

      if (data.documents && Array.isArray(data.documents)) {
        setDocuments(data.documents);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load dashboard metrics");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
  }, [selectedDocFilter, dateFilter]);

  return (
    <div className="dashboard-container">
      {/* HEADER & FILTERS */}
      <div className="dashboard-header-bar">
        <div>
          <h2 className="dashboard-title">Trust-Aware System Analytics</h2>
          <p className="dashboard-subtitle">
            Empirical evaluation metrics derived strictly from actual stored document & query execution history
          </p>
        </div>

        {/* CONTROLS */}
        <div className="dashboard-filter-controls">
          <div className="filter-group">
            <label htmlFor="doc-filter-select">Document Filter:</label>
            <select
              id="doc-filter-select"
              className="dashboard-select"
              value={selectedDocFilter}
              onChange={(e) => setSelectedDocFilter(e.target.value)}
            >
              <option value="all">All Documents ({documents.length})</option>
              {documents.map((d) => (
                <option key={d.document_id} value={d.document_id}>
                  {d.filename || d.document_id}
                </option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <label htmlFor="date-filter-select">Time Range:</label>
            <select
              id="date-filter-select"
              className="dashboard-select"
              value={dateFilter}
              onChange={(e) => setDateFilter(e.target.value)}
            >
              <option value="all">All History</option>
              <option value="today">Today</option>
              <option value="7days">Last 7 Days</option>
              <option value="30days">Last 30 Days</option>
            </select>
          </div>

          <button
            type="button"
            className="dashboard-refresh-btn"
            onClick={fetchStats}
            title="Refresh stored metrics"
          >
            ↻ Refresh
          </button>
        </div>
      </div>

      {loading && (
        <div className="dashboard-loading-state">
          <span className="loader" />
          <p>Loading actual system metrics from storage...</p>
        </div>
      )}

      {error && !loading && (
        <div className="dashboard-error-box">
          <strong>Dashboard Error:</strong> {error}
        </div>
      )}

      {/* EMPTY DATA STATE REQUIREMENT */}
      {!loading && stats && (!stats.has_data || stats.total_questions === 0) && (
        <div className="dashboard-empty-card">
          <span className="empty-chart-icon">📊</span>
          <h3>No data available yet.</h3>
          <p>
            No queries have been executed yet for the selected document and date filter.
            As you upload documents and ask questions, real-time trust evaluations, verification
            outcomes, and latency metrics will be persistently stored and visualized here.
          </p>
        </div>
      )}

      {/* ACTUAL METRICS CONTENT */}
      {!loading && stats && stats.has_data && stats.total_questions > 0 && (
        <div className="dashboard-content-stack">
          {/* SECTION A: HISTORICAL AGGREGATE INFORMATION */}
          <section className="dashboard-section" aria-label="Historical Aggregate Information">
            <div className="section-title-strip">
              <span className="badge-aggregate">Historical Aggregate Information</span>
              <span className="section-note">Based on {stats.total_questions} stored query executions</span>
            </div>

            {/* 11 CORE METRIC CARDS */}
            <div className="metrics-cards-grid">
              {/* 1. Total Documents */}
              <div className="stat-card">
                <span className="stat-num-id">01</span>
                <span className="stat-title">Total Documents</span>
                <strong className="stat-value">{stats.total_documents}</strong>
                <span className="stat-hint">Indexed in FAISS & storage</span>
              </div>

              {/* 2. Total Questions */}
              <div className="stat-card">
                <span className="stat-num-id">02</span>
                <span className="stat-title">Total Questions</span>
                <strong className="stat-value">{stats.total_questions}</strong>
                <span className="stat-hint">Recorded query evaluations</span>
              </div>

              {/* 3. Average Trust Score */}
              <div className="stat-card">
                <span className="stat-num-id">03</span>
                <span className="stat-title">Average Trust Score</span>
                <strong className="stat-value stat-highlight">
                  {stats.average_trust_score != null
                    ? `${Math.round(stats.average_trust_score * 100)}%`
                    : "—"}
                </strong>
                <span className="stat-hint">Calibrated ML & formula score</span>
              </div>

              {/* 4. High-Trust Responses */}
              <div className="stat-card">
                <span className="stat-num-id">04</span>
                <span className="stat-title">High-Trust Responses</span>
                <strong className="stat-value text-emerald">{stats.high_trust_responses}</strong>
                <span className="stat-hint">Score ≥ 75% (Synthesis authorized)</span>
              </div>

              {/* 5. Medium-Trust Responses */}
              <div className="stat-card">
                <span className="stat-num-id">05</span>
                <span className="stat-title">Medium-Trust Responses</span>
                <strong className="stat-value text-amber">{stats.medium_trust_responses}</strong>
                <span className="stat-hint">50% ≤ Score &lt; 75% (Expansion loop)</span>
              </div>

              {/* 6. Low-Trust/Abstained Responses */}
              <div className="stat-card">
                <span className="stat-num-id">06</span>
                <span className="stat-title">Low-Trust / Abstained</span>
                <strong className="stat-value text-crimson">
                  {stats.low_trust_abstained_responses}
                </strong>
                <span className="stat-hint">Score &lt; 50% or safety refusal</span>
              </div>

              {/* 7. Verified Answers */}
              <div className="stat-card">
                <span className="stat-num-id">07</span>
                <span className="stat-title">Verified Answers</span>
                <strong className="stat-value text-emerald">{stats.verified_answers}</strong>
                <span className="stat-hint">All claims confirmed by evidence</span>
              </div>

              {/* 8. Unsupported Answers */}
              <div className="stat-card">
                <span className="stat-num-id">08</span>
                <span className="stat-title">Unsupported Answers</span>
                <strong className="stat-value text-crimson">{stats.unsupported_answers}</strong>
                <span className="stat-hint">Failed claim verification</span>
              </div>

              {/* 9. Contradictions Detected */}
              <div className="stat-card">
                <span className="stat-num-id">09</span>
                <span className="stat-title">Contradictions Detected</span>
                <strong className="stat-value text-amber">
                  {stats.contradictions_detected}
                </strong>
                <span className="stat-hint">Factual conflict pairs flagged</span>
              </div>

              {/* 10. Average Retrieval Relevance */}
              <div className="stat-card">
                <span className="stat-num-id">10</span>
                <span className="stat-title">Avg Retrieval Relevance</span>
                <strong className="stat-value">
                  {stats.average_retrieval_relevance != null
                    ? `${Math.round(stats.average_retrieval_relevance * 100)}%`
                    : "—"}
                </strong>
                <span className="stat-hint">Critic passage relevance score</span>
              </div>

              {/* 11. Average Response Time */}
              <div className="stat-card">
                <span className="stat-num-id">11</span>
                <span className="stat-title">Avg Response Time</span>
                <strong className="stat-value">
                  {stats.average_response_time_ms != null
                    ? `${(stats.average_response_time_ms / 1000).toFixed(2)}s`
                    : "—"}
                </strong>
                <span className="stat-hint">Full retrieval & verification latency</span>
              </div>
            </div>

            {/* SIMPLE CHARTS */}
            <div className="dashboard-charts-grid">
              {/* Chart 1: Trust Level Distribution */}
              <div className="chart-box">
                <h4 className="chart-heading">Trust Level Distribution</h4>
                <div className="distribution-bars">
                  {stats.trust_distribution?.map((item) => {
                    const pct =
                      stats.total_questions > 0
                        ? Math.round((item.count / stats.total_questions) * 100)
                        : 0;
                    return (
                      <div className="dist-row" key={item.level}>
                        <div className="dist-labels">
                          <span className="dist-name">{item.name}</span>
                          <span className="dist-val">
                            {item.count} ({pct}%)
                          </span>
                        </div>
                        <div className="dist-bar-track">
                          <div
                            className={`dist-bar-fill level-${item.level.toLowerCase()}`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Chart 2: Verification Outcomes */}
              <div className="chart-box">
                <h4 className="chart-heading">Verification Outcomes</h4>
                <div className="distribution-bars">
                  {stats.verification_distribution?.map((item, idx) => {
                    const pct =
                      stats.total_questions > 0
                        ? Math.round((item.count / stats.total_questions) * 100)
                        : 0;
                    return (
                      <div className="dist-row" key={idx}>
                        <div className="dist-labels">
                          <span className="dist-name">{item.name}</span>
                          <span className="dist-val">
                            {item.count} ({pct}%)
                          </span>
                        </div>
                        <div className="dist-bar-track">
                          <div
                            className={`dist-bar-fill vstatus-${item.status.toLowerCase()}`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* HISTORICAL QUERIES TABLE */}
            {stats.queries && stats.queries.length > 0 && (
              <div className="history-table-box">
                <h4 className="chart-heading">Stored Execution History</h4>
                <div className="table-responsive">
                  <table className="dashboard-table">
                    <thead>
                      <tr>
                        <th>Timestamp</th>
                        <th>Question</th>
                        <th>Document</th>
                        <th>Trust Score</th>
                        <th>Trust Level</th>
                        <th>Verification</th>
                        <th>Latency</th>
                        <th>Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {stats.queries.map((q) => (
                        <tr
                          key={q.query_id}
                          className={inspectedQuery?.query_id === q.query_id ? "row-selected" : ""}
                        >
                          <td className="mono-cell">
                            {q.timestamp ? q.timestamp.replace("T", " ").slice(0, 19) : "—"}
                          </td>
                          <td className="question-cell" title={q.question}>
                            {q.question}
                          </td>
                          <td className="doc-cell" title={q.document_id}>
                            {q.document_id}
                          </td>
                          <td className="mono-cell">
                            {q.trust_score != null ? `${Math.round(q.trust_score * 100)}%` : "—"}
                          </td>
                          <td>
                            <span className={`pill-small level-${q.trust_level.toLowerCase()}`}>
                              {q.trust_level}
                            </span>
                          </td>
                          <td>
                            <span
                              className={`pill-small v-${q.verification_status.toLowerCase()}`}
                            >
                              {q.verification_status}
                            </span>
                          </td>
                          <td className="mono-cell">{q.response_time_ms}ms</td>
                          <td>
                            <button
                              type="button"
                              className="inspect-btn"
                              onClick={() => setInspectedQuery(q)}
                            >
                              Inspect
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </section>

          {/* SECTION B: CURRENT QUERY INFORMATION */}
          <section className="dashboard-section" aria-label="Current Query Information">
            <div className="section-title-strip">
              <span className="badge-current">Current Query Information</span>
              <span className="section-note">
                Detailed telemetry for inspected query (strictly distinguished from aggregate history)
              </span>
            </div>

            {inspectedQuery ? (
              <div className="inspected-query-card">
                <div className="inspected-query-header">
                  <div>
                    <span className="inspected-label">Evaluated Question:</span>
                    <h3 className="inspected-question">"{inspectedQuery.question}"</h3>
                  </div>
                  <div className="inspected-meta-tags">
                    <span className="mono-tag">Doc: {inspectedQuery.document_id}</span>
                    <span className="mono-tag">
                      Latency: {inspectedQuery.response_time_ms} ms
                    </span>
                  </div>
                </div>

                <div className="inspected-body-grid">
                  <div className="inspected-answer-pane">
                    <span className="pane-subheading">Generated Response</span>
                    <p className="inspected-answer-text">
                      {inspectedQuery.is_abstention
                        ? "I don't have enough reliable evidence in the uploaded document to answer this question."
                        : inspectedQuery.answer ||
                          "No text recorded for this query."}
                    </p>
                  </div>

                  <div className="inspected-stats-pane">
                    <span className="pane-subheading">Verification Signals</span>
                    <div className="inspected-metrics-list">
                      <div className="inspected-metric-row">
                        <span>Trust Score:</span>
                        <strong>
                          {inspectedQuery.trust_score != null
                            ? `${Math.round(inspectedQuery.trust_score * 100)}%`
                            : "—"}
                        </strong>
                      </div>
                      <div className="inspected-metric-row">
                        <span>Trust Level:</span>
                        <span
                          className={`pill-small level-${inspectedQuery.trust_level?.toLowerCase()}`}
                        >
                          {inspectedQuery.trust_level}
                        </span>
                      </div>
                      <div className="inspected-metric-row">
                        <span>Verification Status:</span>
                        <span
                          className={`pill-small v-${inspectedQuery.verification_status?.toLowerCase()}`}
                        >
                          {inspectedQuery.verification_status}
                        </span>
                      </div>
                      <div className="inspected-metric-row">
                        <span>Contradictions:</span>
                        <span>{inspectedQuery.contradictions_count || 0} detected</span>
                      </div>
                      <div className="inspected-metric-row">
                        <span>Avg Relevance:</span>
                        <span>
                          {inspectedQuery.average_relevance != null
                            ? `${Math.round(inspectedQuery.average_relevance * 100)}%`
                            : "—"}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="inspected-placeholder">
                <p>
                  Click <strong>Inspect</strong> on any row in the execution history table above,
                  or ask a question in Chat, to view detailed query telemetry here.
                </p>
              </div>
            )}
          </section>

          {/* DECOUPLING / SAFETY NOTICE */}
          <footer className="dashboard-safety-footer">
            <span className="safety-shield">🛡</span>
            <div>
              <strong>Strict Evidence Isolation Guarantee:</strong>
              <p>
                Dashboard telemetry and query execution history are stored exclusively for administrative
                monitoring, empirical auditing, and research evaluations. Historical dashboard records
                are never injected into vector retrieval indices or provided as factual evidence for
                answering user questions.
              </p>
            </div>
          </footer>
        </div>
      )}
    </div>
  );
}
