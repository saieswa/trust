"""Tests for the Baseline RAG vs. Trust-Aware Multi-Agent RAG comparative evaluation framework."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.evaluation.benchmark_dataset import BENCHMARK_CASES
from app.evaluation.evaluator import run_comparative_evaluation
from app.main import app


def test_benchmark_dataset_contains_all_five_categories() -> None:
    """Verify that the test dataset contains cases for all 5 required stress categories."""
    categories = {c["category"] for c in BENCHMARK_CASES}
    expected = {
        "answerable",
        "not_answerable",
        "ambiguous",
        "conflicting",
        "hallucination_testing",
    }
    assert expected.issubset(categories)
    assert len(BENCHMARK_CASES) >= 10


def test_comparative_evaluation_measures_seven_dimensions() -> None:
    """Verify that comparative evaluation calculates all 7 core dimensions without fabricated numbers."""
    result = run_comparative_evaluation(persist=False)
    assert "summary_table" in result
    assert "category_table" in result
    assert "cases" in result
    assert result["total_cases"] == len(BENCHMARK_CASES)

    metrics_names = [row["metric"] for row in result["summary_table"]]
    assert "Answer Faithfulness" in metrics_names
    assert "Hallucination / Unsupported Claim Rate" in metrics_names
    assert "Retrieval Relevance" in metrics_names
    assert "Refusal / Abstention Appropriateness" in metrics_names
    assert "Contradiction Detection Rate" in metrics_names
    assert "Verification Accuracy" in metrics_names
    assert "Response Latency" in metrics_names

    # Verify Trust-Aware superiority on safety metrics
    row_map = {row["metric"]: row for row in result["summary_table"]}

    # Faithfulness should be higher in Trust-Aware
    ta_faith = int(row_map["Answer Faithfulness"]["trust_aware"].replace("%", ""))
    base_faith = int(row_map["Answer Faithfulness"]["baseline"].replace("%", ""))
    assert ta_faith > base_faith

    # Hallucination rate should be significantly lower in Trust-Aware
    ta_halluc = int(row_map["Hallucination / Unsupported Claim Rate"]["trust_aware"].replace("%", ""))
    base_halluc = int(row_map["Hallucination / Unsupported Claim Rate"]["baseline"].replace("%", ""))
    assert ta_halluc < base_halluc

    # Contradiction detection must be detected by Trust-Aware
    ta_contra = int(row_map["Contradiction Detection Rate"]["trust_aware"].replace("%", ""))
    assert ta_contra == 100


def test_comparative_evaluation_api_endpoints() -> None:
    """Verify GET /api/evaluation/comparative and POST /api/evaluation/run."""
    client = TestClient(app)

    # POST run
    res_run = client.post("/api/evaluation/run")
    assert res_run.status_code == 200
    data_run = res_run.json()
    assert "summary_table" in data_run
    assert len(data_run["summary_table"]) == 7

    # GET comparative
    res_get = client.get("/api/evaluation/comparative")
    assert res_get.status_code == 200
    data_get = res_get.json()
    assert "category_table" in data_get
    assert len(data_get["category_table"]) == 5
    assert "preliminary_notice" in data_get
    assert "nine_point_comparison" in data_get

    # GET comparison-report (9-point architecture comparison)
    res_nine = client.get("/api/evaluation/comparison-report")
    assert res_nine.status_code == 200
    data_nine = res_nine.json()
    assert "dimensions" in data_nine
    assert len(data_nine["dimensions"]) == 9
    dim_ids = [d["id"] for d in data_nine["dimensions"]]
    assert "retrieval" in dim_ids
    assert "evidence_checking" in dim_ids
    assert "contradiction_detection" in dim_ids
    assert "trust_score" in dim_ids
    assert "decision_making" in dim_ids
    assert "abstention" in dim_ids
    assert "answer_verification" in dim_ids
    assert "hallucination_handling" in dim_ids
    assert "evidence_transparency" in dim_ids

    # Verify tradeoffs and limitations disclosures
    assert "tradeoffs_and_limitations" in data_nine
    tradeoffs = data_nine["tradeoffs_and_limitations"]
    assert "latency_and_cost" in tradeoffs
    assert tradeoffs["latency_and_cost"]["winner"] == "Normal RAG"
    assert "inconclusive_areas" in tradeoffs
