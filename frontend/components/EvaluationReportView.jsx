"use client";

import { useEffect, useMemo, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export default function EvaluationReportView() {
  // Navigation & View State
  const [activeMainTab, setActiveMainTab] = useState("datasets"); // "datasets" | "stress_tests"
  const [activeDatasetTab, setActiveDatasetTab] = useState("all"); // "all" | "HaluEval" | "TruthfulQA" | "FEVER" | "HotpotQA"
  const [activeStressSubView, setActiveStressSubView] = useState("review2"); // "review2" | "benchmark" | "cases"
  
  // Data State
  const [datasetData, setDatasetData] = useState(null);
  const [stressData, setStressData] = useState(null);
  const [statusData, setStatusData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  // Execution & Progress State
  const [runningEval, setRunningEval] = useState(false);
  const [runningStress, setRunningStress] = useState(false);
  const [evalProgress, setEvalProgress] = useState({ progress: 0, step: "IDLE", message: "" });
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [configLimits, setConfigLimits] = useState({
    HaluEval: 10,
    TruthfulQA: 10,
    FEVER: 10,
    HotpotQA: 10,
  });
  const [configSeed, setConfigSeed] = useState(42);

  // Case Matrix Filter State
  const [stressCategoryFilter, setStressCategoryFilter] = useState("all");
  const [expandedCaseId, setExpandedCaseId] = useState(null);

  // Fetch Dataset Benchmark Results
  const fetchDatasetResults = async () => {
    try {
      const res = await fetch(`${API_URL}/api/evaluation/dataset/results`);
      if (res.ok) {
        const json = await res.json();
        setDatasetData(json);
      }
    } catch (err) {
      console.warn("Dataset results fetch note:", err);
    }
  };

  // Fetch Status & Availability
  const fetchStatus = async () => {
    try {
      const res = await fetch(`${API_URL}/api/evaluation/dataset/status`);
      if (res.ok) {
        const json = await res.json();
        setStatusData(json);
      }
    } catch (err) {
      console.warn("Status fetch note:", err);
    }
  };

  // Fetch Preserved Stress Tests Data
  const fetchStressData = async () => {
    try {
      const res = await fetch(`${API_URL}/api/evaluation/comparative`);
      if (res.ok) {
        const json = await res.json();
        setStressData(json);
      }
    } catch (err) {
      console.warn("Stress data fetch note:", err);
    }
  };

  const loadAll = async () => {
    setLoading(true);
    setError("");
    try {
      await Promise.allSettled([
        fetchDatasetResults(),
        fetchStatus(),
        fetchStressData(),
      ]);
    } catch (err) {
      setError("Failed to load evaluation environment.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAll();
  }, []);

  // Poll for live execution progress
  useEffect(() => {
    let timer = null;
    if (runningEval) {
      timer = setInterval(async () => {
        try {
          const res = await fetch(`${API_URL}/api/evaluation/dataset/progress`);
          if (res.ok) {
            const state = await res.json();
            setEvalProgress(state);
            if (!state.is_running && state.step === "COMPLETE") {
              setRunningEval(false);
              fetchDatasetResults();
              fetchStatus();
            } else if (!state.is_running && state.step === "FAILED") {
              setRunningEval(false);
              setError(state.message || "Evaluation execution failed.");
            }
          }
        } catch (e) {
          // ignore transient poll errors
        }
      }, 1500);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [runningEval]);

  // Run Dataset Evaluation
  const handleRunDatasetEvaluation = async () => {
    setRunningEval(true);
    setError("");
    setShowConfigModal(false);
    setEvalProgress({ progress: 5, step: "STARTING", message: "Starting dataset evaluation pipeline..." });
    try {
      const res = await fetch(`${API_URL}/api/evaluation/dataset/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          limits: configLimits,
          seed: configSeed,
          top_k: 3,
        }),
      });
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    } catch (err) {
      setRunningEval(false);
      setError(err instanceof Error ? err.message : "Failed to trigger evaluation.");
    }
  };

  // Re-run Deterministic Stress Tests
  const handleRunStressEvaluation = async () => {
    setRunningStress(true);
    setError("");
    try {
      const res = await fetch(`${API_URL}/api/evaluation/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      const json = await res.json();
      // Set data directly from the POST response (contains enriched category_table)
      setStressData(json);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to run stress benchmark.");
    } finally {
      setRunningStress(false);
    }
  };

  // Derived Values from Real Dataset API
  const meta = datasetData?.evaluation_metadata || {};
  const overallNormal = datasetData?.overall?.normal_rag || {};
  const overallTrust = datasetData?.overall?.trust_aware || {};
  const calibration = datasetData?.calibration || { ece: 0, brier_score: 0, bins: [] };
  const datasetsMap = datasetData?.datasets || {};

  const totalEvaluatedN = meta.total_cases || 0;
  const isLiveLLM = meta.evaluation_mode === "live_llm";
  const timestampStr = meta.timestamp ? new Date(meta.timestamp).toLocaleString() : "Recent Run";

  const stressCases = stressData?.cases || [];
  const filteredStressCases = useMemo(() => {
    if (stressCategoryFilter === "all") return stressCases;
    return stressCases.filter((c) => c.category === stressCategoryFilter);
  }, [stressCases, stressCategoryFilter]);

  if (loading && !datasetData && !stressData) {
    return (
      <div className="eval-loading-state">
        <div className="eval-spinner"></div>
        <p>Loading empirical evaluation benchmarks...</p>
      </div>
    );
  }

  return (
    <div className="eval-report-container">
      {/* TOP HEADER BAR */}
      <div className="eval-header-bar">
        <div>
          <div className="eval-title-badge">Empirical Comparative Evaluation</div>
          <h2 className="eval-main-title">Dataset Evaluation</h2>
          <p className="eval-main-subtitle">
            Evaluation of Normal RAG and Trust-Aware RAG across public factuality and reasoning benchmarks.
          </p>
        </div>

        <div className="eval-actions-group">
          {/* Status Badge */}
          <div
            className="eval-mode-badge"
            style={{
              display: "flex",
              alignItems: "center",
              gap: "6px",
              padding: "6px 12px",
              borderRadius: "6px",
              fontSize: "12px",
              fontWeight: 600,
              backgroundColor: isLiveLLM ? "#e6f4ea" : "#fef7e0",
              color: isLiveLLM ? "#137333" : "#b06000",
              border: `1px solid ${isLiveLLM ? "#ceead6" : "#fce8b2"}`,
            }}
          >
            <span style={{ fontSize: "10px" }}>{isLiveLLM ? "🟢" : "🟠"}</span>
            <span>{isLiveLLM ? "LIVE LLM EVALUATION" : "DETERMINISTIC FALLBACK EVALUATION"}</span>
          </div>

          {/* Timestamp Badge */}
          <div className="eval-timestamp-badge">
            <span className="eval-clock-icon">🕒</span> {timestampStr}
          </div>

          {/* Config Limits Button */}
          <button
            type="button"
            className="eval-config-btn"
            style={{
              padding: "8px 12px",
              background: "#ffffff",
              border: "1px solid #d1d5db",
              borderRadius: "6px",
              cursor: "pointer",
              fontWeight: 600,
              fontSize: "13px",
              color: "#374151",
            }}
            onClick={() => setShowConfigModal((v) => !v)}
          >
            ⚙️ Configure Limits
          </button>

          {/* Run Button */}
          <button
            type="button"
            className="eval-run-btn"
            onClick={handleRunDatasetEvaluation}
            disabled={runningEval}
            style={{ minWidth: "210px" }}
          >
            {runningEval ? (
              <>
                <span className="eval-mini-spinner"></span> Running Pipeline...
              </>
            ) : (
              <>⚡ Run Dataset Evaluation</>
            )}
          </button>
        </div>
      </div>

      {/* CONFIGURATION POPUP */}
      {showConfigModal && (
        <div
          style={{
            background: "#ffffff",
            border: "1px solid #e5e7eb",
            borderRadius: "10px",
            padding: "16px 20px",
            marginBottom: "20px",
            boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.1)",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
            <h4 style={{ margin: 0, fontSize: "15px", fontWeight: 700, color: "#111827" }}>
              Configure Dataset Evaluation Limits
            </h4>
            <button
              onClick={() => setShowConfigModal(false)}
              style={{ background: "none", border: "none", cursor: "pointer", fontSize: "16px" }}
            >
              ✕
            </button>
          </div>
          <p style={{ margin: "0 0 14px 0", fontSize: "13px", color: "#6b7280" }}>
            Select reproducible evaluation sample sizes across each official dataset:
          </p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "12px" }}>
            {["HaluEval", "TruthfulQA", "FEVER", "HotpotQA"].map((ds) => (
              <div key={ds} style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                <label style={{ fontSize: "12px", fontWeight: 600, color: "#374151" }}>{ds} Cases:</label>
                <input
                  type="number"
                  min="1"
                  max="100"
                  value={configLimits[ds]}
                  onChange={(e) =>
                    setConfigLimits({ ...configLimits, [ds]: Math.max(1, parseInt(e.target.value) || 1) })
                  }
                  style={{
                    padding: "6px 10px",
                    border: "1px solid #d1d5db",
                    borderRadius: "6px",
                    fontSize: "13px",
                  }}
                />
              </div>
            ))}
            <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
              <label style={{ fontSize: "12px", fontWeight: 600, color: "#374151" }}>Sampling Seed:</label>
              <input
                type="number"
                value={configSeed}
                onChange={(e) => setConfigSeed(parseInt(e.target.value) || 42)}
                style={{
                  padding: "6px 10px",
                  border: "1px solid #d1d5db",
                  borderRadius: "6px",
                  fontSize: "13px",
                }}
              />
            </div>
          </div>
          <div style={{ marginTop: "14px", display: "flex", justifyContent: "flex-end", gap: "8px" }}>
            <button
              onClick={() => {
                setConfigLimits({ HaluEval: 25, TruthfulQA: 25, FEVER: 20, HotpotQA: 20 });
                setConfigSeed(42);
              }}
              style={{
                padding: "6px 12px",
                fontSize: "12px",
                background: "#f3f4f6",
                border: "1px solid #d1d5db",
                borderRadius: "6px",
                cursor: "pointer",
              }}
            >
              Reset to Recommended (90 Cases)
            </button>
            <button
              onClick={handleRunDatasetEvaluation}
              style={{
                padding: "6px 14px",
                fontSize: "12px",
                background: "#2563eb",
                color: "#ffffff",
                border: "none",
                borderRadius: "6px",
                fontWeight: 600,
                cursor: "pointer",
              }}
            >
              Apply & Run
            </button>
          </div>
        </div>
      )}

      {/* LIVE PROGRESS BAR */}
      {runningEval && (
        <div
          style={{
            background: "#eff6ff",
            border: "1px solid #bfdbfe",
            borderRadius: "8px",
            padding: "12px 16px",
            marginBottom: "20px",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
            <span style={{ fontSize: "13px", fontWeight: 600, color: "#1e40af" }}>
              {evalProgress.message || "Running evaluation pipeline..."}
            </span>
            <span style={{ fontSize: "12px", fontWeight: 700, color: "#1e40af" }}>
              {evalProgress.progress}%
            </span>
          </div>
          <div style={{ width: "100%", height: "8px", background: "#dbeafe", borderRadius: "999px", overflow: "hidden" }}>
            <div
              style={{
                width: `${evalProgress.progress}%`,
                height: "100%",
                background: "#2563eb",
                transition: "width 0.3s ease",
              }}
            ></div>
          </div>
        </div>
      )}

      {/* ERROR ALERT */}
      {error && (
        <div
          style={{
            background: "#fee2e2",
            border: "1px solid #fca5a5",
            borderRadius: "8px",
            padding: "10px 14px",
            marginBottom: "16px",
            color: "#991b1b",
            fontSize: "13px",
            display: "flex",
            alignItems: "center",
            gap: "8px",
          }}
        >
          <span>⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {/* MAIN TOP-LEVEL NAVIGATION BAR */}
      <div className="eval-subnav-bar" style={{ marginBottom: "22px" }}>
        <button
          type="button"
          className={`eval-subnav-pill ${activeMainTab === "datasets" ? "is-active" : ""}`}
          onClick={() => setActiveMainTab("datasets")}
        >
          📊 Dataset Evaluation (N = {totalEvaluatedN} Cases)
        </button>
        <button
          type="button"
          className={`eval-subnav-pill ${activeMainTab === "stress_tests" ? "is-active" : ""}`}
          onClick={() => setActiveMainTab("stress_tests")}
        >
          🧪 Deterministic Stress Tests (Regression Suite)
        </button>
      </div>

      {/* ========================================================================= */}
      {/* VIEW A: DATASET EVALUATION (PRIMARY) */}
      {/* ========================================================================= */}
      {activeMainTab === "datasets" && (
        <>
          {/* DATASET SUMMARY CARDS (SECTION 11) */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
              gap: "16px",
              marginBottom: "24px",
            }}
          >
            {/* HaluEval Card */}
            <div className="eval-kpi-card" style={{ borderTop: "4px solid #3b82f6" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                <span style={{ fontSize: "14px", fontWeight: 700, color: "#1e3a8a" }}>HaluEval</span>
                <span style={{ fontSize: "11px", background: "#dbeafe", color: "#1e40af", padding: "2px 6px", borderRadius: "4px" }}>
                  N = {datasetsMap?.HaluEval?.cases ?? 0}
                </span>
              </div>
              <div style={{ fontSize: "12px", color: "#6b7280", marginBottom: "12px" }}>
                Hallucination Detection Benchmark
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: "6px", fontSize: "13px" }}>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "#4b5563" }}>Accuracy:</span>
                  <span>
                    <strong style={{ color: "#6b7280" }}>{datasetsMap?.HaluEval?.normal_rag?.accuracy ?? 0}%</strong> →{" "}
                    <strong style={{ color: "#15803d" }}>{datasetsMap?.HaluEval?.trust_aware?.accuracy ?? 0}%</strong>
                  </span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "#4b5563" }}>Hallucination:</span>
                  <span>
                    <span style={{ color: "#dc2626" }}>{datasetsMap?.HaluEval?.normal_rag?.hallucination_rate ?? 0}%</span> →{" "}
                    <span style={{ color: "#15803d" }}>{datasetsMap?.HaluEval?.trust_aware?.hallucination_rate ?? 0}%</span>
                  </span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "#4b5563" }}>Abstention:</span>
                  <strong style={{ color: "#2563eb" }}>{datasetsMap?.HaluEval?.trust_aware?.abstention_rate ?? 0}%</strong>
                </div>
              </div>
            </div>

            {/* TruthfulQA Card */}
            <div className="eval-kpi-card" style={{ borderTop: "4px solid #10b981" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                <span style={{ fontSize: "14px", fontWeight: 700, color: "#065f46" }}>TruthfulQA</span>
                <span style={{ fontSize: "11px", background: "#d1fae5", color: "#065f46", padding: "2px 6px", borderRadius: "4px" }}>
                  N = {datasetsMap?.TruthfulQA?.cases ?? 0}
                </span>
              </div>
              <div style={{ fontSize: "12px", color: "#6b7280", marginBottom: "12px" }}>
                Misconceptions & Truthfulness
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: "6px", fontSize: "13px" }}>
                {/* Truthfulness: Normal RAG vs Trust-Aware (among answered only) */}
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "#4b5563" }}>Truthfulness (Normal):</span>
                  <strong style={{ color: "#6b7280" }}>
                    {datasetsMap?.TruthfulQA?.truthfulqa_metrics?.normal_rag_truthfulness
                      ?? datasetsMap?.TruthfulQA?.normal_rag?.accuracy
                      ?? 0}%
                  </strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "#4b5563" }}>Truthfulness (Answered):</span>
                  <strong style={{ color: "#15803d" }}>
                    {datasetsMap?.TruthfulQA?.truthfulqa_metrics?.truthfulness_among_answered
                      ?? datasetsMap?.TruthfulQA?.trust_aware?.accuracy
                      ?? 0}%
                  </strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "#4b5563" }}>Overall Accuracy:</span>
                  <strong style={{ color: "#1d4ed8" }}>
                    {datasetsMap?.TruthfulQA?.truthfulqa_metrics?.overall_accuracy_incl_abstentions
                      ?? datasetsMap?.TruthfulQA?.trust_aware?.accuracy
                      ?? 0}%
                  </strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "#4b5563" }}>Misconception Rate:</span>
                  <span>
                    <span style={{ color: "#dc2626" }}>{datasetsMap?.TruthfulQA?.normal_rag?.hallucination_rate ?? 0}%</span> →{" "}
                    <span style={{ color: "#15803d" }}>{datasetsMap?.TruthfulQA?.truthfulqa_metrics?.hallucination_rate ?? datasetsMap?.TruthfulQA?.trust_aware?.hallucination_rate ?? 0}%</span>
                  </span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "#4b5563" }}>Abstention:</span>
                  <strong style={{ color: "#2563eb" }}>{datasetsMap?.TruthfulQA?.truthfulqa_metrics?.abstention_rate ?? datasetsMap?.TruthfulQA?.trust_aware?.abstention_rate ?? 0}%</strong>
                </div>
                {datasetsMap?.TruthfulQA?.truthfulqa_metrics && (
                  <div style={{ display: "flex", justifyContent: "space-between", borderTop: "1px dashed #e5e7eb", paddingTop: "4px", marginTop: "2px" }}>
                    <span style={{ color: "#9ca3af", fontSize: "11px" }}>
                      Answered: {datasetsMap.TruthfulQA.truthfulqa_metrics.n_answered} / {datasetsMap.TruthfulQA.truthfulqa_metrics.n_total} &nbsp;|&nbsp; Abstained: {datasetsMap.TruthfulQA.truthfulqa_metrics.n_abstained}
                    </span>
                  </div>
                )}
              </div>
            </div>


            {/* FEVER Card */}
            <div className="eval-kpi-card" style={{ borderTop: "4px solid #8b5cf6" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                <span style={{ fontSize: "14px", fontWeight: 700, color: "#4c1d95" }}>FEVER</span>
                <span style={{ fontSize: "11px", background: "#ede9fe", color: "#5b21b6", padding: "2px 6px", borderRadius: "4px" }}>
                  N = {datasetsMap?.FEVER?.cases ?? 0}
                </span>
              </div>
              <div style={{ fontSize: "12px", color: "#6b7280", marginBottom: "12px" }}>
                Fact Extraction & Verification
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: "6px", fontSize: "13px" }}>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "#4b5563" }}>Accuracy:</span>
                  <span>
                    <strong style={{ color: "#6b7280" }}>{datasetsMap?.FEVER?.normal_rag?.accuracy ?? 0}%</strong> →{" "}
                    <strong style={{ color: "#15803d" }}>{datasetsMap?.FEVER?.trust_aware?.accuracy ?? 0}%</strong>
                  </span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "#4b5563" }}>Macro F1:</span>
                  <span>
                    <span style={{ color: "#6b7280" }}>{datasetsMap?.FEVER?.fever_metrics?.normal?.macro_f1 ? (datasetsMap.FEVER.fever_metrics.normal.macro_f1 * 100).toFixed(1) : 0}%</span> →{" "}
                    <span style={{ color: "#15803d" }}>{datasetsMap?.FEVER?.fever_metrics?.trust_aware?.macro_f1 ? (datasetsMap.FEVER.fever_metrics.trust_aware.macro_f1 * 100).toFixed(1) : 0}%</span>
                  </span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "#4b5563" }}>Abstention:</span>
                  <strong style={{ color: "#2563eb" }}>{datasetsMap?.FEVER?.trust_aware?.abstention_rate ?? 0}%</strong>
                </div>
              </div>
            </div>

            {/* HotpotQA Card */}
            <div className="eval-kpi-card" style={{ borderTop: "4px solid #f59e0b" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                <span style={{ fontSize: "14px", fontWeight: 700, color: "#78350f" }}>HotpotQA</span>
                <span style={{ fontSize: "11px", background: "#fef3c7", color: "#92400e", padding: "2px 6px", borderRadius: "4px" }}>
                  N = {datasetsMap?.HotpotQA?.cases ?? 0}
                </span>
              </div>
              <div style={{ fontSize: "12px", color: "#6b7280", marginBottom: "12px" }}>
                Multi-Hop Reasoning Benchmark
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: "6px", fontSize: "13px" }}>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "#4b5563" }}>Token F1:</span>
                  <span>
                    <strong style={{ color: "#6b7280" }}>{datasetsMap?.HotpotQA?.normal_rag?.f1 ?? 0}%</strong> →{" "}
                    <strong style={{ color: "#15803d" }}>{datasetsMap?.HotpotQA?.trust_aware?.f1 ?? 0}%</strong>
                  </span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "#4b5563" }}>Exact Match (EM):</span>
                  <span>
                    <span style={{ color: "#6b7280" }}>{datasetsMap?.HotpotQA?.hotpot_metrics?.normal_em ?? 0}%</span> →{" "}
                    <span style={{ color: "#15803d" }}>{datasetsMap?.HotpotQA?.hotpot_metrics?.trust_em ?? 0}%</span>
                  </span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "#4b5563" }}>Accuracy:</span>
                  <strong style={{ color: "#15803d" }}>{datasetsMap?.HotpotQA?.trust_aware?.accuracy ?? 0}%</strong>
                </div>
              </div>
            </div>
          </div>

          {/* ACCURACY COMPARISON CHART */}
          <section className="eval-section-card" style={{ marginBottom: "26px" }}>
            <div className="eval-section-header">
              <div>
                <h3 className="eval-section-title">Accuracy Comparison</h3>
                <p className="eval-section-subtitle">
                  Empirical head-to-head accuracy benchmarking Normal RAG (Baseline) against Trust-Aware RAG.
                </p>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px", color: "#4b5563" }}>
                  <span style={{ display: "inline-block", width: "12px", height: "12px", borderRadius: "3px", background: "#94a3b8" }}></span>
                  <span>Normal RAG (Baseline)</span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px", color: "#1e40af", fontWeight: 600 }}>
                  <span style={{ display: "inline-block", width: "12px", height: "12px", borderRadius: "3px", background: "#2563eb" }}></span>
                  <span>Trust-Aware RAG</span>
                </div>
              </div>
            </div>

            {/* CHART SVG CONTAINER */}
            <div style={{ padding: "20px 16px 12px 16px", background: "#f9fafb", borderRadius: "8px", border: "1px solid #e5e7eb" }}>
              <div style={{ width: "100%", maxWidth: "780px", margin: "0 auto" }}>
                {(() => {
                  const comparisonItems = [
                    {
                      label: "Overall",
                      normal: Number(overallNormal.accuracy ?? 0),
                      trust: Number(overallTrust.accuracy ?? 0),
                      isOverall: true,
                    },
                    {
                      label: "HaluEval",
                      normal: Number(datasetsMap?.HaluEval?.normal_rag?.accuracy ?? 0),
                      trust: Number(datasetsMap?.HaluEval?.trust_aware?.accuracy ?? 0),
                    },
                    {
                      label: "TruthfulQA",
                      normal: Number(datasetsMap?.TruthfulQA?.normal_rag?.accuracy ?? 0),
                      trust: Number(datasetsMap?.TruthfulQA?.trust_aware?.accuracy ?? 0),
                    },
                    {
                      label: "FEVER",
                      normal: Number(datasetsMap?.FEVER?.normal_rag?.accuracy ?? 0),
                      trust: Number(datasetsMap?.FEVER?.trust_aware?.accuracy ?? 0),
                    },
                    {
                      label: "HotpotQA",
                      normal: Number(datasetsMap?.HotpotQA?.normal_rag?.accuracy ?? 0),
                      trust: Number(datasetsMap?.HotpotQA?.trust_aware?.accuracy ?? 0),
                    },
                  ];

                  const svgWidth = 740;
                  const svgHeight = 270;
                  const chartTop = 35;
                  const chartBottom = 220;
                  const chartHeight = chartBottom - chartTop;
                  const chartLeft = 55;
                  const chartRight = 710;
                  const chartWidth = chartRight - chartLeft;
                  const groupWidth = chartWidth / comparisonItems.length;
                  const barWidth = 26;
                  const barGap = 6;

                  return (
                    <svg
                      viewBox={`0 0 ${svgWidth} ${svgHeight}`}
                      style={{ width: "100%", height: "auto", overflow: "visible" }}
                      role="img"
                      aria-label="Accuracy Comparison Chart"
                    >
                      {/* Grid Lines & Y-Axis Labels */}
                      {[0, 25, 50, 75, 100].map((tick) => {
                        const y = chartBottom - (tick / 100) * chartHeight;
                        return (
                          <g key={tick}>
                            <line
                              x1={chartLeft}
                              y1={y}
                              x2={chartRight}
                              y2={y}
                              stroke="#e5e7eb"
                              strokeDasharray={tick === 0 ? "none" : "3 3"}
                              strokeWidth={tick === 0 ? "1.5" : "1"}
                            />
                            <text
                              x={chartLeft - 10}
                              y={y + 4}
                              textAnchor="end"
                              fontSize="11"
                              fill="#6b7280"
                              fontWeight="500"
                            >
                              {tick}%
                            </text>
                          </g>
                        );
                      })}

                      {/* X Axis Baseline */}
                      <line
                        x1={chartLeft}
                        y1={chartBottom}
                        x2={chartRight}
                        y2={chartBottom}
                        stroke="#9ca3af"
                        strokeWidth="1.5"
                      />

                      {/* Bar Groups */}
                      {comparisonItems.map((item, idx) => {
                        const groupCenterX = chartLeft + (idx + 0.5) * groupWidth;
                        const normalBarX = groupCenterX - barWidth - barGap / 2;
                        const trustBarX = groupCenterX + barGap / 2;

                        const normalBarHeight = Math.max(2, (item.normal / 100) * chartHeight);
                        const trustBarHeight = Math.max(2, (item.trust / 100) * chartHeight);

                        const normalBarY = chartBottom - normalBarHeight;
                        const trustBarY = chartBottom - trustBarHeight;

                        const delta = item.trust - item.normal;
                        const isPositiveDelta = delta >= 0;

                        return (
                          <g key={item.label}>
                            {/* Subtle group separator / background highlight for Overall */}
                            {item.isOverall && (
                              <rect
                                x={groupCenterX - groupWidth / 2 + 6}
                                y={chartTop - 10}
                                width={groupWidth - 12}
                                height={chartHeight + 10}
                                fill="#eff6ff"
                                opacity="0.4"
                                rx="6"
                              />
                            )}

                            {/* Delta Badge above bars */}
                            <g>
                              <rect
                                x={groupCenterX - 24}
                                y={Math.min(normalBarY, trustBarY) - 24}
                                width="48"
                                height="17"
                                rx="4"
                                fill={isPositiveDelta ? "#dcfce7" : "#fee2e2"}
                                stroke={isPositiveDelta ? "#86efac" : "#fca5a5"}
                                strokeWidth="0.8"
                              />
                              <text
                                x={groupCenterX}
                                y={Math.min(normalBarY, trustBarY) - 12}
                                textAnchor="middle"
                                fontSize="10"
                                fontWeight="700"
                                fill={isPositiveDelta ? "#15803d" : "#dc2626"}
                              >
                                {isPositiveDelta ? `+${delta.toFixed(1)}%` : `${delta.toFixed(1)}%`}
                              </text>
                            </g>

                            {/* Normal RAG Bar */}
                            <rect
                              x={normalBarX}
                              y={normalBarY}
                              width={barWidth}
                              height={normalBarHeight}
                              fill="#94a3b8"
                              rx="3"
                            >
                              <title>{`Normal RAG (${item.label}): ${item.normal}%`}</title>
                            </rect>
                            <text
                              x={normalBarX + barWidth / 2}
                              y={normalBarY - 4}
                              textAnchor="middle"
                              fontSize="10"
                              fontWeight="600"
                              fill="#475569"
                            >
                              {item.normal}%
                            </text>

                            {/* Trust-Aware RAG Bar */}
                            <rect
                              x={trustBarX}
                              y={trustBarY}
                              width={barWidth}
                              height={trustBarHeight}
                              fill="#2563eb"
                              rx="3"
                            >
                              <title>{`Trust-Aware RAG (${item.label}): ${item.trust}%`}</title>
                            </rect>
                            <text
                              x={trustBarX + barWidth / 2}
                              y={trustBarY - 4}
                              textAnchor="middle"
                              fontSize="10"
                              fontWeight="700"
                              fill="#1d4ed8"
                            >
                              {item.trust}%
                            </text>

                            {/* X-Axis Category Label */}
                            <text
                              x={groupCenterX}
                              y={chartBottom + 20}
                              textAnchor="middle"
                              fontSize="12"
                              fontWeight={item.isOverall ? "700" : "600"}
                              fill={item.isOverall ? "#1e40af" : "#374151"}
                            >
                              {item.label}
                            </text>
                          </g>
                        );
                      })}
                    </svg>
                  );
                })()}
              </div>
            </div>
          </section>

          {/* OVERALL COMPARISON TABLE (SECTION 12) */}
          <section className="eval-section-card" style={{ marginBottom: "26px" }}>
            <div className="eval-section-header">
              <div>
                <h3 className="eval-section-title">Overall Performance Comparison</h3>
                <p className="eval-section-subtitle">
                  Empirically measured head-to-head metrics on identical benchmark questions and context.
                </p>
              </div>
              <div style={{ fontSize: "12px", color: "#6b7280", background: "#f3f4f6", padding: "4px 8px", borderRadius: "4px" }}>
                Total Cases Evaluated: <strong>N = {totalEvaluatedN}</strong>
              </div>
            </div>

            <div className="eval-table-wrapper">
              <table className="eval-data-table">
                <thead>
                  <tr>
                    <th style={{ width: "26%" }}>Metric</th>
                    <th style={{ width: "24%" }}>Normal RAG (Baseline)</th>
                    <th style={{ width: "24%" }}>Trust-Aware RAG</th>
                    <th style={{ width: "26%" }}>Empirical Impact & Significance</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><strong>Accuracy</strong></td>
                    <td className="txt-bold">{overallNormal.accuracy ?? 0}%</td>
                    <td className="txt-bold" style={{ color: (overallTrust.accuracy ?? 0) >= (overallNormal.accuracy ?? 0) ? "#15803d" : "#b45309" }}>
                      {overallTrust.accuracy ?? 0}%
                    </td>
                    <td>
                      <span className="eval-delta-tag tag-good">
                        {((overallTrust.accuracy ?? 0) - (overallNormal.accuracy ?? 0)).toFixed(1)}%
                      </span>{" "}
                      Task correctness cleared against official ground truth
                    </td>
                  </tr>
                  <tr>
                    <td><strong>Precision</strong></td>
                    <td>{overallNormal.precision ?? 0}%</td>
                    <td style={{ color: "#15803d", fontWeight: 600 }}>{overallTrust.precision ?? 0}%</td>
                    <td>Token-level answer precision of statements made</td>
                  </tr>
                  <tr>
                    <td><strong>Recall</strong></td>
                    <td>{overallNormal.recall ?? 0}%</td>
                    <td style={{ color: "#15803d", fontWeight: 600 }}>{overallTrust.recall ?? 0}%</td>
                    <td>Fraction of required ground-truth information covered</td>
                  </tr>
                  <tr>
                    <td><strong>F1 Score</strong></td>
                    <td className="txt-bold">{overallNormal.f1 ?? 0}%</td>
                    <td className="txt-bold" style={{ color: "#15803d" }}>{overallTrust.f1 ?? 0}%</td>
                    <td>Harmonic mean of token precision and recall</td>
                  </tr>
                  <tr>
                    <td><strong>Hallucination Rate</strong></td>
                    <td style={{ color: "#dc2626", fontWeight: 600 }}>{overallNormal.hallucination_rate ?? 0}%</td>
                    <td style={{ color: "#15803d", fontWeight: 700 }}>{overallTrust.hallucination_rate ?? 0}%</td>
                    <td>
                      <span className="eval-delta-tag tag-good">
                        {((overallTrust.hallucination_rate ?? 0) - (overallNormal.hallucination_rate ?? 0)).toFixed(1)}%
                      </span>{" "}
                      Unverified / incorrect claims caught by Verifier
                    </td>
                  </tr>
                  <tr>
                    <td><strong>Abstention Rate</strong></td>
                    <td>{overallNormal.abstention_rate ?? 0}% (Never abstains)</td>
                    <td style={{ color: "#2563eb", fontWeight: 600 }}>{overallTrust.abstention_rate ?? 0}%</td>
                    <td>Honest refusal when trust &lt; 0.40 or contradictory</td>
                  </tr>
                  <tr>
                    <td><strong>Retrieval Relevance</strong></td>
                    <td>{overallNormal.retrieval_relevance ?? 0}%</td>
                    <td>{overallTrust.retrieval_relevance ?? 0}%</td>
                    <td>Identical isolated FAISS top-k retrieval conditions</td>
                  </tr>
                  <tr>
                    <td><strong>Response Latency</strong></td>
                    <td style={{ color: "#15803d", fontWeight: 600 }}>{overallNormal.latency_ms ?? 0} ms</td>
                    <td style={{ color: "#4b5563" }}>{overallTrust.latency_ms ?? 0} ms</td>
                    <td>
                      <span className="eval-delta-tag tag-neutral">Trade-Off</span>{" "}
                      Multi-agent verification overhead for strict safety
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>

          {/* DATASET-WISE COMPARISON TABS (SECTION 13) */}
          <section className="eval-section-card" style={{ marginBottom: "26px" }}>
            <div className="eval-section-header">
              <div>
                <h3 className="eval-section-title">Dataset-Wise Detailed Breakdown</h3>
                <p className="eval-section-subtitle">
                  Inspect task-specific metrics for each evaluated benchmark dataset.
                </p>
              </div>
            </div>

            {/* Sub-tabs */}
            <div style={{ display: "flex", gap: "8px", borderBottom: "1px solid #e5e7eb", paddingBottom: "12px", marginBottom: "16px" }}>
              {["all", "HaluEval", "TruthfulQA", "FEVER", "HotpotQA"].map((tab) => (
                <button
                  key={tab}
                  type="button"
                  onClick={() => setActiveDatasetTab(tab)}
                  style={{
                    padding: "6px 14px",
                    borderRadius: "6px",
                    fontSize: "13px",
                    fontWeight: 600,
                    cursor: "pointer",
                    border: "none",
                    background: activeDatasetTab === tab ? "#2563eb" : "#f3f4f6",
                    color: activeDatasetTab === tab ? "#ffffff" : "#4b5563",
                  }}
                >
                  {tab === "all" ? "All Benchmarks Summary" : tab}
                </button>
              ))}
            </div>

            {/* TAB CONTENT: ALL */}
            {activeDatasetTab === "all" && (
              <div className="eval-table-wrapper">
                <table className="eval-data-table">
                  <thead>
                    <tr>
                      <th>Benchmark</th>
                      <th>Cases (N)</th>
                      <th>Primary Target</th>
                      <th>Normal RAG Accuracy</th>
                      <th>Trust-Aware Accuracy</th>
                      <th>Normal Hallucination</th>
                      <th>Trust Hallucination</th>
                      <th>Trust Abstention</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(datasetsMap).map(([name, d]) => (
                      <tr key={name}>
                        <td><strong>{name}</strong></td>
                        <td>{d.cases ?? 0}</td>
                        <td>
                          {name === "HaluEval" && "Hallucination Rejection"}
                          {name === "TruthfulQA" && "Truth vs Misconception"}
                          {name === "FEVER" && "3-Way Fact Verification"}
                          {name === "HotpotQA" && "Multi-Hop Reasoning QA"}
                        </td>
                        <td>{d.normal_rag?.accuracy ?? 0}%</td>
                        <td style={{ color: "#15803d", fontWeight: 700 }}>{d.trust_aware?.accuracy ?? 0}%</td>
                        <td style={{ color: "#dc2626" }}>{d.normal_rag?.hallucination_rate ?? 0}%</td>
                        <td style={{ color: "#15803d", fontWeight: 600 }}>{d.trust_aware?.hallucination_rate ?? 0}%</td>
                        <td>{d.trust_aware?.abstention_rate ?? 0}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* TAB CONTENT: FEVER */}
            {activeDatasetTab === "FEVER" && (
              <div>
                <div style={{ marginBottom: "12px", fontSize: "13px", color: "#4b5563" }}>
                  FEVER evaluates 3-class fact verification (<strong>SUPPORTS</strong>, <strong>REFUTES</strong>, <strong>NOT ENOUGH INFO</strong>).
                  Macro-averaged metrics ensure each category is evaluated with equal importance.
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px", marginBottom: "16px" }}>
                  <div style={{ background: "#f9fafb", padding: "14px", borderRadius: "8px", border: "1px solid #e5e7eb" }}>
                    <h4 style={{ margin: "0 0 8px 0", fontSize: "14px", color: "#374151" }}>Normal RAG (Baseline)</h4>
                    <div style={{ fontSize: "13px", display: "flex", flexDirection: "column", gap: "4px" }}>
                      <div>Accuracy: <strong>{datasetsMap?.FEVER?.normal_rag?.accuracy ?? 0}%</strong></div>
                      <div>Macro F1: <strong>{datasetsMap?.FEVER?.fever_metrics?.normal?.macro_f1 ? (datasetsMap.FEVER.fever_metrics.normal.macro_f1 * 100).toFixed(1) : 0}%</strong></div>
                      <div>Macro Precision: <strong>{datasetsMap?.FEVER?.fever_metrics?.normal?.macro_precision ? (datasetsMap.FEVER.fever_metrics.normal.macro_precision * 100).toFixed(1) : 0}%</strong></div>
                      <div>Macro Recall: <strong>{datasetsMap?.FEVER?.fever_metrics?.normal?.macro_recall ? (datasetsMap.FEVER.fever_metrics.normal.macro_recall * 100).toFixed(1) : 0}%</strong></div>
                      <div style={{ color: "#6b7280", marginTop: "4px", fontSize: "12px" }}>
                        * Normal RAG lacks contradiction classification; defaults to adopting retrieved claim.
                      </div>
                    </div>
                  </div>

                  <div style={{ background: "#f0fdf4", padding: "14px", borderRadius: "8px", border: "1px solid #bbf7d0" }}>
                    <h4 style={{ margin: "0 0 8px 0", fontSize: "14px", color: "#166534" }}>Trust-Aware RAG</h4>
                    <div style={{ fontSize: "13px", display: "flex", flexDirection: "column", gap: "4px" }}>
                      <div>Accuracy: <strong style={{ color: "#15803d" }}>{datasetsMap?.FEVER?.trust_aware?.accuracy ?? 0}%</strong></div>
                      <div>Macro F1: <strong style={{ color: "#15803d" }}>{datasetsMap?.FEVER?.fever_metrics?.trust_aware?.macro_f1 ? (datasetsMap.FEVER.fever_metrics.trust_aware.macro_f1 * 100).toFixed(1) : 0}%</strong></div>
                      <div>Macro Precision: <strong style={{ color: "#15803d" }}>{datasetsMap?.FEVER?.fever_metrics?.trust_aware?.macro_precision ? (datasetsMap.FEVER.fever_metrics.trust_aware.macro_precision * 100).toFixed(1) : 0}%</strong></div>
                      <div>Macro Recall: <strong style={{ color: "#15803d" }}>{datasetsMap?.FEVER?.fever_metrics?.trust_aware?.macro_recall ? (datasetsMap.FEVER.fever_metrics.trust_aware.macro_recall * 100).toFixed(1) : 0}%</strong></div>
                      <div style={{ color: "#166534", marginTop: "4px", fontSize: "12px" }}>
                        * Actively routes contradictory evidence to REFUTES and insufficient evidence to NOT ENOUGH INFO.
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* TAB CONTENT: HotpotQA */}
            {activeDatasetTab === "HotpotQA" && (
              <div>
                <div style={{ marginBottom: "12px", fontSize: "13px", color: "#4b5563" }}>
                  HotpotQA tests multi-hop question answering across multiple context paragraphs.
                  Evaluated using official SQuAD / HotpotQA Token Precision, Recall, F1, and Exact Match (EM).
                </div>
                <div className="eval-table-wrapper">
                  <table className="eval-data-table">
                    <thead>
                      <tr>
                        <th>Metric</th>
                        <th>Normal RAG</th>
                        <th>Trust-Aware RAG</th>
                        <th>Description</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td><strong>Token F1 Score</strong></td>
                        <td>{datasetsMap?.HotpotQA?.normal_rag?.f1 ?? 0}%</td>
                        <td style={{ color: "#15803d", fontWeight: 700 }}>{datasetsMap?.HotpotQA?.trust_aware?.f1 ?? 0}%</td>
                        <td>Token overlap with ground truth target</td>
                      </tr>
                      <tr>
                        <td><strong>Exact Match (EM)</strong></td>
                        <td>{datasetsMap?.HotpotQA?.hotpot_metrics?.normal_em ?? 0}%</td>
                        <td style={{ color: "#15803d", fontWeight: 700 }}>{datasetsMap?.HotpotQA?.hotpot_metrics?.trust_em ?? 0}%</td>
                        <td>Exact normalized answer string match</td>
                      </tr>
                      <tr>
                        <td><strong>Accuracy</strong></td>
                        <td>{datasetsMap?.HotpotQA?.normal_rag?.accuracy ?? 0}%</td>
                        <td style={{ color: "#15803d", fontWeight: 700 }}>{datasetsMap?.HotpotQA?.trust_aware?.accuracy ?? 0}%</td>
                        <td>F1 &gt;= 0.40 threshold or substring containment</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* TAB CONTENT: TruthfulQA */}
            {activeDatasetTab === "TruthfulQA" && (
              <div>
                <div style={{ marginBottom: "12px", fontSize: "13px", color: "#4b5563" }}>
                  TruthfulQA tests whether system answers are truthful or adopt popular human misconceptions.
                  Evaluated against official best answers, correct answer sets, and incorrect misconception sets.
                </div>
                <div className="eval-table-wrapper">
                  <table className="eval-data-table">
                    <thead>
                      <tr>
                        <th>Dimension</th>
                        <th>Normal RAG</th>
                        <th>Trust-Aware RAG</th>
                        <th>Significance</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td><strong>Truthfulness — Normal RAG</strong></td>
                        <td style={{ color: "#6b7280" }}>{datasetsMap?.TruthfulQA?.truthfulqa_metrics?.normal_rag_truthfulness ?? datasetsMap?.TruthfulQA?.normal_rag?.accuracy ?? 0}%</td>
                        <td>—</td>
                        <td>Answers aligning with verifiable facts (all N cases)</td>
                      </tr>
                      <tr>
                        <td><strong>Truthfulness — Answered Only</strong></td>
                        <td>—</td>
                        <td style={{ color: "#15803d", fontWeight: 700 }}>
                          {datasetsMap?.TruthfulQA?.truthfulqa_metrics?.truthfulness_among_answered ?? datasetsMap?.TruthfulQA?.trust_aware?.accuracy ?? 0}%
                        </td>
                        <td>Factual accuracy among cases where system answered (excludes abstentions)</td>
                      </tr>
                      <tr>
                        <td><strong>Overall Accuracy (incl. abstentions)</strong></td>
                        <td>—</td>
                        <td style={{ color: "#1d4ed8", fontWeight: 700 }}>
                          {datasetsMap?.TruthfulQA?.truthfulqa_metrics?.overall_accuracy_incl_abstentions ?? datasetsMap?.TruthfulQA?.trust_aware?.accuracy ?? 0}%
                        </td>
                        <td>Truthful / N_total. Abstentions count as 0 (conservative measure)</td>
                      </tr>
                      <tr>
                        <td><strong>Misconception Rate</strong></td>
                        <td style={{ color: "#dc2626" }}>{datasetsMap?.TruthfulQA?.normal_rag?.hallucination_rate ?? 0}%</td>
                        <td style={{ color: "#15803d", fontWeight: 700 }}>{datasetsMap?.TruthfulQA?.truthfulqa_metrics?.hallucination_rate ?? datasetsMap?.TruthfulQA?.trust_aware?.hallucination_rate ?? 0}%</td>
                        <td>Fell into popular human misconception answer</td>
                      </tr>
                      <tr>
                        <td><strong>Abstention Rate</strong></td>
                        <td>0.0%</td>
                        <td style={{ color: "#2563eb", fontWeight: 600 }}>{datasetsMap?.TruthfulQA?.truthfulqa_metrics?.abstention_rate ?? datasetsMap?.TruthfulQA?.trust_aware?.abstention_rate ?? 0}%</td>
                        <td>Cases where Trust-Aware refused to answer (insufficient evidence)</td>
                      </tr>
                      <tr style={{ background: "#f9fafb" }}>
                        <td><strong>Answered / Abstained</strong></td>
                        <td>All {datasetsMap?.TruthfulQA?.cases ?? 0} answered</td>
                        <td style={{ color: "#374151" }}>
                          {datasetsMap?.TruthfulQA?.truthfulqa_metrics
                            ? `${datasetsMap.TruthfulQA.truthfulqa_metrics.n_answered} answered, ${datasetsMap.TruthfulQA.truthfulqa_metrics.n_abstained} abstained`
                            : "—"}
                        </td>
                        <td>Abstentions are honest refusals, not wrong answers</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            )}


            {/* TAB CONTENT: HaluEval */}
            {activeDatasetTab === "HaluEval" && (
              <div>
                <div style={{ marginBottom: "12px", fontSize: "13px", color: "#4b5563" }}>
                  HaluEval benchmark evaluates factual accuracy versus known fabricated hallucinated answers.
                </div>
                <div className="eval-table-wrapper">
                  <table className="eval-data-table">
                    <thead>
                      <tr>
                        <th>Metric</th>
                        <th>Normal RAG</th>
                        <th>Trust-Aware RAG</th>
                        <th>Observation</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td><strong>Factual Correctness</strong></td>
                        <td>{datasetsMap?.HaluEval?.normal_rag?.accuracy ?? 0}%</td>
                        <td style={{ color: "#15803d", fontWeight: 700 }}>{datasetsMap?.HaluEval?.trust_aware?.accuracy ?? 0}%</td>
                        <td>Generating the verified right answer</td>
                      </tr>
                      <tr>
                        <td><strong>Hallucination Rate</strong></td>
                        <td style={{ color: "#dc2626" }}>{datasetsMap?.HaluEval?.normal_rag?.hallucination_rate ?? 0}%</td>
                        <td style={{ color: "#15803d", fontWeight: 700 }}>{datasetsMap?.HaluEval?.trust_aware?.hallucination_rate ?? 0}%</td>
                        <td>Matching the known hallucinated target</td>
                      </tr>
                      <tr>
                        <td><strong>Abstention Rate</strong></td>
                        <td>0.0%</td>
                        <td style={{ color: "#2563eb", fontWeight: 600 }}>{datasetsMap?.HaluEval?.trust_aware?.abstention_rate ?? 0}%</td>
                        <td>Abstaining when context is inconclusive</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </section>

          {/* TRUST CALIBRATION SECTION (SECTION 14) */}
          <section className="eval-section-card" style={{ marginBottom: "26px" }}>
            <div className="eval-section-header">
              <div>
                <h3 className="eval-section-title">Trust Calibration</h3>
                <p className="eval-section-subtitle">
                  Expected Calibration Error (ECE) and Brier Score measuring reliability between predicted trust scores and actual correctness.
                </p>
              </div>
              <div style={{ display: "flex", gap: "12px" }}>
                <div style={{ background: "#f0fdf4", border: "1px solid #bbf7d0", padding: "6px 12px", borderRadius: "6px" }}>
                  <span style={{ fontSize: "11px", color: "#166534" }}>ECE (Target: &lt; 0.15):</span>{" "}
                  <strong style={{ fontSize: "14px", color: "#15803d" }}>{calibration.ece ?? 0}</strong>
                </div>
                <div style={{ background: "#eff6ff", border: "1px solid #bfdbfe", padding: "6px 12px", borderRadius: "6px" }}>
                  <span style={{ fontSize: "11px", color: "#1e40af" }}>Brier Score (Target: &lt; 0.20):</span>{" "}
                  <strong style={{ fontSize: "14px", color: "#1e40af" }}>{calibration.brier_score ?? 0}</strong>
                </div>
              </div>
            </div>

            {/* RELIABILITY CHART (PREDICTED TRUST VS ACTUAL ACCURACY) */}
            <div style={{ padding: "16px", background: "#f9fafb", borderRadius: "8px", border: "1px solid #e5e7eb", marginBottom: "16px" }}>
              <div style={{ fontSize: "13px", fontWeight: 700, color: "#374151", marginBottom: "10px" }}>
                Reliability Diagram: Predicted Trust Score vs. Empirical Accuracy (10 Confidence Bins)
              </div>
              
              {/* SVG Chart */}
              <div style={{ width: "100%", maxWidth: "600px", margin: "0 auto" }}>
                <svg viewBox="0 0 500 260" style={{ width: "100%", height: "auto", overflow: "visible" }}>
                  {/* Grid lines */}
                  {[0, 0.25, 0.5, 0.75, 1.0].map((v, idx) => {
                    const y = 220 - v * 180;
                    return (
                      <g key={idx}>
                        <line x1="50" y1={y} x2="470" y2={y} stroke="#e5e7eb" strokeDasharray="3 3" />
                        <text x="40" y={y + 4} textAnchor="end" fontSize="10" fill="#6b7280">
                          {Math.round(v * 100)}%
                        </text>
                      </g>
                    );
                  })}

                  {/* Perfect calibration diagonal */}
                  <line x1="50" y1="220" x2="470" y2="40" stroke="#9ca3af" strokeWidth="2" strokeDasharray="4 4" />
                  <text x="440" y="32" fontSize="9" fill="#6b7280" fontWeight="600">Ideal Calibration (y = x)</text>

                  {/* Axes */}
                  <line x1="50" y1="220" x2="470" y2="220" stroke="#4b5563" strokeWidth="1.5" />
                  <line x1="50" y1="40" x2="50" y2="220" stroke="#4b5563" strokeWidth="1.5" />

                  {/* X Axis labels */}
                  {[0, 0.2, 0.4, 0.6, 0.8, 1.0].map((v, idx) => {
                    const x = 50 + v * 420;
                    return (
                      <g key={idx}>
                        <line x1={x} y1="220" x2={x} y2="224" stroke="#4b5563" />
                        <text x={x} y="238" textAnchor="middle" fontSize="10" fill="#6b7280">
                          {v.toFixed(1)}
                        </text>
                      </g>
                    );
                  })}

                  {/* Axis titles */}
                  <text x="260" y="255" textAnchor="middle" fontSize="11" fontWeight="600" fill="#374151">
                    Mean Predicted Trust Score (Confidence)
                  </text>
                  <text x="-130" y="15" transform="rotate(-90)" textAnchor="middle" fontSize="11" fontWeight="600" fill="#374151">
                    Empirical Accuracy
                  </text>

                  {/* Bins bars / points */}
                  {calibration.bins?.map((b, idx) => {
                    if (b.count === 0 || b.accuracy === null) return null;
                    const x = 50 + b.mean_confidence * 420;
                    const y = 220 - b.accuracy * 180;
                    return (
                      <g key={idx}>
                        {/* Bar */}
                        <rect
                          x={x - 8}
                          y={y}
                          width="16"
                          height={220 - y}
                          fill="#3b82f6"
                          opacity="0.65"
                          rx="2"
                        />
                        {/* Dot */}
                        <circle cx={x} cy={y} r="4" fill="#1d4ed8" stroke="#ffffff" strokeWidth="1.5" />
                        <text x={x} y={y - 8} textAnchor="middle" fontSize="9" fontWeight="700" fill="#1e40af">
                          {(b.accuracy * 100).toFixed(0)}%
                        </text>
                      </g>
                    );
                  })}
                </svg>
              </div>

              {/* Bins Table */}
              <div style={{ marginTop: "14px", overflowX: "auto" }}>
                <table style={{ width: "100%", fontSize: "12px", textAlign: "left", borderCollapse: "collapse" }}>
                  <thead>
                    <tr style={{ background: "#f3f4f6", borderBottom: "1px solid #e5e7eb" }}>
                      <th style={{ padding: "6px 10px" }}>Bin Range</th>
                      <th style={{ padding: "6px 10px" }}>Cases</th>
                      <th style={{ padding: "6px 10px" }}>Mean Predicted Trust</th>
                      <th style={{ padding: "6px 10px" }}>Empirical Accuracy</th>
                      <th style={{ padding: "6px 10px" }}>Calibration Gap</th>
                    </tr>
                  </thead>
                  <tbody>
                    {calibration.bins?.map((b, idx) => {
                      const gap = b.accuracy !== null ? Math.abs(b.mean_confidence - b.accuracy).toFixed(3) : "-";
                      return (
                        <tr key={idx} style={{ borderBottom: "1px solid #f3f4f6" }}>
                          <td style={{ padding: "5px 10px" }}>[{b.bin_range[0].toFixed(1)} - {b.bin_range[1].toFixed(1)}]</td>
                          <td style={{ padding: "5px 10px" }}>{b.count}</td>
                          <td style={{ padding: "5px 10px" }}>{b.count > 0 ? b.mean_confidence.toFixed(3) : "-"}</td>
                          <td style={{ padding: "5px 10px" }}>{b.accuracy !== null ? `${(b.accuracy * 100).toFixed(1)}%` : "-"}</td>
                          <td style={{ padding: "5px 10px", color: gap !== "-" && parseFloat(gap) > 0.15 ? "#dc2626" : "#15803d" }}>
                            {gap}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </section>
        </>
      )}

      {/* ========================================================================= */}
      {/* VIEW B: DETERMINISTIC STRESS TESTS (PRESERVED REGRESSION SUITE) */}
      {/* ========================================================================= */}
      {activeMainTab === "stress_tests" && (
        <>
          <div className="eval-notice-banner" style={{ marginBottom: "20px" }}>
            <span className="eval-notice-icon">🧪</span>
            <div className="eval-notice-content">
              <strong>Deterministic Stress Tests (10 Curated Adversarial Cases)</strong>
              <p>
                Preserved regression suite testing 5 specific edge-case scenarios: Incomplete evidence, Outdated claims, Direct contradiction, Misleading evidence, and Cross-document contamination.
              </p>
            </div>
            <button
              onClick={handleRunStressEvaluation}
              disabled={runningStress}
              style={{
                padding: "6px 14px",
                background: runningStress ? "#e5e7eb" : "#ffffff",
                border: "1px solid #d1d5db",
                borderRadius: "6px",
                fontWeight: 600,
                fontSize: "12px",
                cursor: runningStress ? "not-allowed" : "pointer",
                whiteSpace: "nowrap",
                display: "flex",
                alignItems: "center",
                gap: "6px",
              }}
            >
              {runningStress ? (
                <><span className="eval-mini-spinner" />Running Stress Tests...</>
              ) : (
                <>⚡ Run Stress Tests</>
              )}
            </button>
          </div>

          {/* Sub-nav for Stress Tests */}
          <div className="eval-subnav-bar" style={{ marginBottom: "18px" }}>
            <button
              type="button"
              className={`eval-subnav-pill ${activeStressSubView === "review2" ? "is-active" : ""}`}
              onClick={() => setActiveStressSubView("review2")}
            >
              🎓 9-Point Architectural Comparison
            </button>
            <button
              type="button"
              className={`eval-subnav-pill ${activeStressSubView === "benchmark" ? "is-active" : ""}`}
              onClick={() => setActiveStressSubView("benchmark")}
            >
              📊 Stress Categories Matrix
            </button>
            <button
              type="button"
              className={`eval-subnav-pill ${activeStressSubView === "cases" ? "is-active" : ""}`}
              onClick={() => setActiveStressSubView("cases")}
            >
              🔍 Case-by-Case Viewer ({stressCases.length} Cases)
            </button>
          </div>

          {/* 9-Point Comparison */}
          {activeStressSubView === "review2" && (
            <section className="eval-section-card">
              <div className="eval-section-header">
                <div>
                  <h3 className="eval-section-title">9-Point Architectural Comparison</h3>
                  <p className="eval-section-subtitle">Comparing structural RAG stages from retrieval to refusal</p>
                </div>
              </div>
              <div className="eval-table-wrapper">
                <table className="eval-data-table eval-nine-table">
                  <thead>
                    <tr>
                      <th style={{ width: "20%" }}>Architectural Dimension</th>
                      <th style={{ width: "24%" }}>Normal RAG Pipeline</th>
                      <th style={{ width: "28%" }}>Trust-Aware Multi-Agent RAG</th>
                      <th style={{ width: "28%" }}>Observed Result</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(stressData?.nine_point_comparison?.dimensions || []).map((dim) => (
                      <tr key={dim.dimension}>
                        <td className="eval-dimension-cell"><strong>{dim.dimension}</strong></td>
                        <td className="eval-arch-cell base-cell">{dim.normal_rag ?? dim.baseline_rag ?? "—"}</td>
                        <td className="eval-arch-cell ta-cell">{dim.trust_aware_rag ?? "—"}</td>
                        <td className="eval-verdict-cell">{dim.observed_result ?? dim.experimental_metric ?? "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {/* Stress Benchmark Table */}
          {activeStressSubView === "benchmark" && (
            <section className="eval-section-card">
              <div className="eval-section-header">
                <div>
                  <h3 className="eval-section-title">Performance Across Stress Categories</h3>
                  <p className="eval-section-subtitle">Evaluating 5 adversarial failure modes</p>
                </div>
                {!stressData && (
                  <div style={{
                    fontSize: "12px", background: "#fef3c7", color: "#92400e",
                    padding: "6px 12px", borderRadius: "6px", border: "1px solid #fde68a",
                  }}>
                    ⚠️ Click "Run Stress Tests" to populate results
                  </div>
                )}
              </div>
              <div className="eval-table-wrapper">
                <table className="eval-data-table">
                  <thead>
                    <tr>
                      <th>Stress Category</th>
                      <th>Cases</th>
                      <th>Normal Hallucination</th>
                      <th>Trust Hallucination</th>
                      <th>Trust Abstention</th>
                      <th>Normal Latency</th>
                      <th>Trust Latency</th>
                    </tr>
                  </thead>
                  <tbody>
                    {!stressData ? (
                      /* No run yet — show placeholder rows for the 5 known categories */
                      [
                        "Answerable from Document",
                        "Not Answerable from Document",
                        "Ambiguous / Incomplete Evidence",
                        "Conflicting Evidence",
                        "Hallucination Testing (False Premise)",
                      ].map((label) => (
                        <tr key={label}>
                          <td><strong>{label}</strong></td>
                          {[...Array(6)].map((_, i) => (
                            <td key={i} style={{ color: "#9ca3af", fontStyle: "italic" }}>Not Run</td>
                          ))}
                        </tr>
                      ))
                    ) : (
                      (stressData.category_table || []).map((cat) => {
                        // Helper: format a numeric metric or fall back to "Not available"
                        const fmtPct = (v) => v != null ? `${v}%` : "Not available";
                        const fmtMs  = (v) => v != null ? `${v} ms` : "Not available";
                        return (
                          <tr key={cat.category}>
                            <td><strong>{cat.category}</strong></td>
                            <td>{cat.cases_count ?? cat.test_cases ?? "—"}</td>
                            <td style={{ color: (cat.baseline_hallucination_rate ?? 0) > 0 ? "#dc2626" : "#374151" }}>
                              {fmtPct(cat.baseline_hallucination_rate)}
                            </td>
                            <td style={{ color: (cat.trust_hallucination_rate ?? 0) === 0 ? "#15803d" : "#b45309", fontWeight: 700 }}>
                              {fmtPct(cat.trust_hallucination_rate)}
                            </td>
                            <td>{fmtPct(cat.trust_abstention_rate)}</td>
                            <td>{fmtMs(cat.baseline_latency_ms)}</td>
                            <td>{fmtMs(cat.trust_latency_ms)}</td>
                          </tr>
                        );
                      })
                    )}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {/* Case Viewer */}
          {activeStressSubView === "cases" && (
            <section className="eval-section-card">
              <div className="eval-section-header">
                <div>
                  <h3 className="eval-section-title">Case-by-Case Inspection Matrix</h3>
                  <p className="eval-section-subtitle">Side-by-side prompt, retrieval, and decision inspector</p>
                </div>
              </div>
              <div className="eval-cases-stream">
                {filteredStressCases.map((c) => {
                  const isExpanded = expandedCaseId === c.case_id;
                  // API returns c.baseline (not c.baseline_rag) and c.trust_aware (not c.trust_aware_rag)
                  const base = c.baseline ?? c.baseline_rag ?? null;
                  const ta = c.trust_aware ?? c.trust_aware_rag ?? null;

                  // Safe formatters — distinguish real 0 from missing
                  const fmtHallucination = (val) => {
                    if (val == null) return "Not available";
                    return val > 0 ? "Hallucinated" : "Grounded";
                  };
                  const fmtTrustScore = (score) => {
                    if (score == null) return "Not available";
                    return `${(score * 100).toFixed(1)}%`;
                  };

                  return (
                    <div key={c.case_id} className={`eval-case-card ${isExpanded ? "is-expanded" : ""}`}>
                      <div
                        className="eval-case-summary-header"
                        onClick={() => setExpandedCaseId(isExpanded ? null : c.case_id)}
                        style={{ cursor: "pointer" }}
                      >
                        <div className="eval-case-title-col">
                          <span className="eval-case-badge">{c.case_id}</span>
                          <span className="eval-case-question">&ldquo;{c.question}&rdquo;</span>
                        </div>
                        <div className="eval-case-quick-meta">
                          <span className="eval-adv-badge">{c.category}</span>
                          <span className="eval-metric-chip">
                            Normal: <strong>{fmtHallucination(base?.hallucination_rate)}</strong>
                          </span>
                          <span className="eval-metric-chip">
                            Trust: <strong>{ta?.abstained ? "Abstained" : (ta ? "Verified" : "Not available")}</strong>
                          </span>
                          <span>{isExpanded ? "▲" : "▼"}</span>
                        </div>
                      </div>

                      {isExpanded && (
                        <div className="eval-case-detail-body" style={{ padding: "16px", borderTop: "1px solid #e5e7eb" }}>
                          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
                            {/* Normal RAG panel */}
                            <div style={{ background: "#f9fafb", padding: "12px", borderRadius: "6px" }}>
                              <h5 style={{ margin: "0 0 6px 0", color: "#374151" }}>Normal RAG Answer</h5>
                              {base ? (
                                <>
                                  <p style={{ fontSize: "13px", margin: "0 0 6px 0" }}>&ldquo;{base.answer ?? "No answer recorded"}&rdquo;</p>
                                  <div style={{ fontSize: "12px", color: "#6b7280", display: "flex", flexDirection: "column", gap: "2px" }}>
                                    <span>Faithfulness: <strong>{base.faithfulness != null ? `${(base.faithfulness * 100).toFixed(0)}%` : "Not available"}</strong></span>
                                    <span>Hallucination Rate: <strong style={{ color: (base.hallucination_rate ?? 0) > 0 ? "#dc2626" : "#15803d" }}>
                                      {base.hallucination_rate != null ? `${(base.hallucination_rate * 100).toFixed(0)}%` : "Not available"}
                                    </strong></span>
                                    <span>Refused: <strong>{base.refused != null ? (base.refused ? "Yes" : "No") : "Not available"}</strong></span>
                                    <span>Latency: <strong>{base.latency_ms != null ? `${base.latency_ms.toFixed(0)} ms` : "Not available"}</strong></span>
                                  </div>
                                </>
                              ) : (
                                <p style={{ fontSize: "13px", color: "#9ca3af", fontStyle: "italic" }}>Normal RAG data not available</p>
                              )}
                            </div>

                            {/* Trust-Aware panel */}
                            <div style={{ background: "#f0fdf4", padding: "12px", borderRadius: "6px" }}>
                              <h5 style={{ margin: "0 0 6px 0", color: "#166534" }}>Trust-Aware Answer</h5>
                              {ta ? (
                                <>
                                  <p style={{ fontSize: "13px", margin: "0 0 6px 0" }}>&ldquo;{ta.answer ?? "No answer recorded"}&rdquo;</p>
                                  <div style={{ fontSize: "12px", color: "#374151", display: "flex", flexDirection: "column", gap: "2px" }}>
                                    <span>Trust Score: <strong>{fmtTrustScore(ta.trust_score)}</strong></span>
                                    <span>Trust Level: <strong>{ta.trust_level ?? "Not available"}</strong></span>
                                    <span>Verification: <strong>{ta.verification_status ?? "Not available"}</strong></span>
                                    <span>Abstained: <strong>{ta.abstained != null ? (ta.abstained ? "Yes" : "No") : "Not available"}</strong></span>
                                    <span>Hallucination Rate: <strong style={{ color: (ta.hallucination_rate ?? 0) > 0 ? "#dc2626" : "#15803d" }}>
                                      {ta.hallucination_rate != null ? `${(ta.hallucination_rate * 100).toFixed(0)}%` : "Not available"}
                                    </strong></span>
                                    <span>Latency: <strong>{ta.latency_ms != null ? `${ta.latency_ms.toFixed(0)} ms` : "Not available"}</strong></span>
                                  </div>
                                </>
                              ) : (
                                <p style={{ fontSize: "13px", color: "#9ca3af", fontStyle: "italic" }}>Trust-aware data not available</p>
                              )}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </section>
          )}
        </>
      )}
    </div>
  );
}
