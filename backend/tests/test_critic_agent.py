"""Comprehensive unit and integration tests for the Critic Agent."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.agents.critic import CriticAgent, get_critic_agent
from app.api import critic as critic_api
from app.main import app


class FakeCompletions:
    def __init__(self, response_json: dict[str, Any]):
        self.response_json = response_json
        self.request = None

    def create(self, **request):
        self.request = request
        content = json.dumps(self.response_json)
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeGroq:
    def __init__(self, response_json: dict[str, Any]):
        self.chat = SimpleNamespace(completions=FakeCompletions(response_json))


REQUIRED_EVALUATION_KEYS = [
    "document_id",
    "chunk_id",
    "relevance",
    "support status",
    "quality assessment",
    "contradiction status",
    "explanation",
]


# ==============================================================================
# 1. Relevant Evidence Test
# ==============================================================================
def test_critic_evaluates_relevant_evidence() -> None:
    fake_response = {
        "evaluations": [
            {
                "chunk_id": "doc-attention::chunk-0001",
                "relevance": True,
                "support_status": "supported",
                "quality_assessment": {
                    "evidence_strength": "high",
                    "source_quality": "high",
                    "potential_outdated": False,
                    "details": "Peer-reviewed foundational research paper excerpt.",
                },
                "contradiction_status": "none",
                "explanation": "Directly explains that the Transformer is based entirely on attention mechanisms.",
            }
        ],
        "contradictions": [],
    }

    critic = CriticAgent(llm_client=FakeGroq(fake_response))
    evidence = [
        {
            "document_id": "doc-attention",
            "chunk_id": "doc-attention::chunk-0001",
            "filename": "attention_is_all_you_need.pdf",
            "page_number": 1,
            "source": "attention_is_all_you_need.pdf#page=1",
            "text": "The Transformer is the first transduction model relying entirely on self-attention.",
        }
    ]

    result = critic.evaluate(
        question="What does the Transformer rely on?",
        evidence=evidence,
        document_id="doc-attention",
    )

    assert result["document_id"] == "doc-attention"
    assert len(result["evaluations"]) == 1
    item = result["evaluations"][0]

    # Verify all 7 required keys are present
    for key in REQUIRED_EVALUATION_KEYS:
        assert key in item, f"Missing required key: '{key}'"

    assert item["chunk_id"] == "doc-attention::chunk-0001"
    assert item["document_id"] == "doc-attention"
    assert item["relevance"] is True
    assert item["support_status"] == "supported"
    assert item["support status"] == "supported"
    assert item["quality_assessment"]["evidence_strength"] == "high"
    assert item["quality_assessment"]["potential_outdated"] is False
    assert item["contradiction_status"] == "none"
    assert "attention" in item["explanation"].lower()
    assert result["summary"]["relevant_chunks"] == 1
    assert result["summary"]["supporting_chunks"] == 1
    assert result["summary"]["has_contradictions"] is False


# ==============================================================================
# 2. Irrelevant Evidence Test
# ==============================================================================
def test_critic_evaluates_irrelevant_evidence() -> None:
    fake_response = {
        "evaluations": [
            {
                "chunk_id": "doc-recipe::chunk-0001",
                "relevance": False,
                "support_status": "unsupported",
                "quality_assessment": {
                    "evidence_strength": "none",
                    "source_quality": "low",
                    "potential_outdated": False,
                    "details": "Recipe instructions with no connection to neural machine translation.",
                },
                "contradiction_status": "none",
                "explanation": "Discusses baking temperature and yeast hydration; completely irrelevant to Transformer architectures.",
            }
        ],
        "contradictions": [],
    }

    critic = CriticAgent(llm_client=FakeGroq(fake_response))
    evidence = [
        {
            "document_id": "doc-recipe",
            "chunk_id": "doc-recipe::chunk-0001",
            "filename": "sourdough_recipe.txt",
            "page_number": 1,
            "source": "sourdough_recipe.txt#page=1",
            "text": "Bake the sourdough loaf at 450 degrees Fahrenheit for 25 minutes with the Dutch oven lid on.",
        }
    ]

    result = critic.evaluate(
        question="How does multi-head attention work in Transformers?",
        evidence=evidence,
        document_id="doc-recipe",
    )

    item = result["evaluations"][0]
    for key in REQUIRED_EVALUATION_KEYS:
        assert key in item

    assert item["relevance"] is False
    assert item["support_status"] == "unsupported"
    assert item["support status"] == "unsupported"
    assert item["quality_assessment"]["evidence_strength"] == "none"
    assert "irrelevant" in item["explanation"].lower()
    assert result["summary"]["relevant_chunks"] == 0
    assert result["summary"]["supporting_chunks"] == 0


# ==============================================================================
# 3. Supporting Evidence Test
# ==============================================================================
def test_critic_evaluates_supporting_evidence() -> None:
    fake_response = {
        "evaluations": [
            {
                "chunk_id": "doc-bert::chunk-0005",
                "relevance": True,
                "support_status": "supported",
                "quality_assessment": {
                    "evidence_strength": "high",
                    "source_quality": "high",
                    "potential_outdated": False,
                    "details": "Conclusive empirical statement directly verifying bidirectional pre-training.",
                },
                "contradiction_status": "none",
                "explanation": "Directly and factually supports the claim that BERT utilizes bidirectional representations.",
            }
        ],
        "contradictions": [],
    }

    critic = CriticAgent(llm_client=FakeGroq(fake_response))
    evidence = [
        {
            "document_id": "doc-bert",
            "chunk_id": "doc-bert::chunk-0005",
            "filename": "bert_paper.pdf",
            "page_number": 2,
            "source": "bert_paper.pdf#page=2",
            "text": "BERT is designed to pre-train deep bidirectional representations from unlabeled text.",
        }
    ]

    result = critic.evaluate(
        question="Does BERT use deep bidirectional representations?",
        evidence=evidence,
        document_id="doc-bert",
    )

    item = result["evaluations"][0]
    for key in REQUIRED_EVALUATION_KEYS:
        assert key in item

    assert item["relevance"] is True
    assert item["support_status"] == "supported"
    assert item["support status"] == "supported"
    assert item["quality_assessment"]["evidence_strength"] == "high"
    assert item["contradiction_status"] == "none"
    assert "supports" in item["explanation"].lower()
    assert result["summary"]["supporting_chunks"] == 1


# ==============================================================================
# 4. Conflicting Evidence Test (Contradiction Detection)
# ==============================================================================
def test_critic_detects_conflicting_evidence() -> None:
    fake_response = {
        "evaluations": [
            {
                "chunk_id": "doc-experiment::chunk-0001",
                "relevance": True,
                "support_status": "supported",
                "quality_assessment": {
                    "evidence_strength": "high",
                    "source_quality": "medium",
                    "potential_outdated": False,
                    "details": "Section 3 experimental table claims learning rate was 0.001.",
                },
                "contradiction_status": "contradiction_detected: conflicts with chunk-0002 on learning rate",
                "explanation": "Claims learning rate is 0.001, contradicting chunk-0002 which reports 0.05.",
            },
            {
                "chunk_id": "doc-experiment::chunk-0002",
                "relevance": True,
                "support_status": "supported",
                "quality_assessment": {
                    "evidence_strength": "high",
                    "source_quality": "medium",
                    "potential_outdated": False,
                    "details": "Section 4 hyperparameter log claims learning rate was strictly 0.05.",
                },
                "contradiction_status": "contradiction_detected: conflicts with chunk-0001 on learning rate",
                "explanation": "Claims learning rate was 0.05, which directly contradicts chunk-0001.",
            },
        ],
        "contradictions": [
            {
                "chunk_id_a": "doc-experiment::chunk-0001",
                "chunk_id_b": "doc-experiment::chunk-0002",
                "claim_a": "Learning rate is 0.001",
                "claim_b": "Learning rate is strictly 0.05",
                "explanation": "The two chunks specify conflicting learning rate hyperparameters for the same model.",
            }
        ],
    }

    critic = CriticAgent(llm_client=FakeGroq(fake_response))
    evidence = [
        {
            "document_id": "doc-experiment",
            "chunk_id": "doc-experiment::chunk-0001",
            "filename": "experiment_report.pdf",
            "page_number": 3,
            "source": "experiment_report.pdf#page=3",
            "text": "The learning rate was initialized to 0.001 for all training epochs.",
        },
        {
            "document_id": "doc-experiment",
            "chunk_id": "doc-experiment::chunk-0002",
            "filename": "experiment_report.pdf",
            "page_number": 4,
            "source": "experiment_report.pdf#page=4",
            "text": "The learning rate was strictly set to 0.05 throughout all runs.",
        },
    ]

    result = critic.evaluate(
        question="What was the learning rate used during model training?",
        evidence=evidence,
        document_id="doc-experiment",
    )

    assert len(result["evaluations"]) == 2
    assert len(result["contradictions"]) == 1
    assert result["summary"]["has_contradictions"] is True

    # Both items must carry all 7 required keys
    for item in result["evaluations"]:
        for key in REQUIRED_EVALUATION_KEYS:
            assert key in item
        assert "contradiction_detected" in item["contradiction_status"].lower()
        assert item["support_status"] == "contradicted"

    conflict = result["contradictions"][0]
    assert conflict["chunk_id_a"] == "doc-experiment::chunk-0001"
    assert conflict["chunk_id_b"] == "doc-experiment::chunk-0002"
    assert "learning rate" in conflict["explanation"].lower()


# ==============================================================================
# 5. Evidence from Another Document Test (Strict Document Isolation)
# ==============================================================================
def test_critic_rejects_evidence_from_another_document() -> None:
    # Fake response only needs to handle the valid chunk from target document
    fake_response = {
        "evaluations": [
            {
                "chunk_id": "doc-valid::chunk-0001",
                "relevance": True,
                "support_status": "supported",
                "quality_assessment": {
                    "evidence_strength": "high",
                    "source_quality": "high",
                    "potential_outdated": False,
                    "details": "Valid document excerpt.",
                },
                "contradiction_status": "none",
                "explanation": "Belongs to the target document and answers the question.",
            }
        ],
        "contradictions": [],
    }

    critic = CriticAgent(llm_client=FakeGroq(fake_response))
    evidence = [
        {
            "document_id": "doc-valid",
            "chunk_id": "doc-valid::chunk-0001",
            "filename": "target_paper.pdf",
            "page_number": 1,
            "source": "target_paper.pdf#page=1",
            "text": "This document introduces the primary neural algorithm.",
        },
        {
            "document_id": "doc-unrelated-foreign",
            "chunk_id": "doc-unrelated-foreign::chunk-9999",
            "filename": "financial_statement_2020.pdf",
            "page_number": 12,
            "source": "financial_statement_2020.pdf#page=12",
            "text": "Operating revenues increased by 14% year over year.",
        },
    ]

    result = critic.evaluate(
        question="What neural algorithm is introduced?",
        evidence=evidence,
        document_id="doc-valid",
    )

    assert len(result["evaluations"]) == 2

    # Find the unrelated document evaluation
    unrelated_item = next(
        e for e in result["evaluations"] if e["chunk_id"] == "doc-unrelated-foreign::chunk-9999"
    )

    for key in REQUIRED_EVALUATION_KEYS:
        assert key in unrelated_item

    assert unrelated_item["document_id"] == "doc-unrelated-foreign"
    assert unrelated_item["relevance"] is False
    assert unrelated_item["support_status"] == "unrelated_document"
    assert unrelated_item["support status"] == "unrelated_document"
    assert unrelated_item["quality_assessment"]["valid_document"] is False
    assert unrelated_item["quality_assessment"]["evidence_strength"] == "none"
    assert unrelated_item["contradiction_status"] == "none"
    assert "unrelated document" in unrelated_item["explanation"].lower()
    assert result["summary"]["unrelated_chunks"] == 1


# ==============================================================================
# 6. Edge Cases & Validation Tests
# ==============================================================================
def test_critic_raises_on_empty_question() -> None:
    critic = CriticAgent(llm_client=FakeGroq({}))
    with pytest.raises(ValueError, match="question must contain non-empty text"):
        critic.evaluate(question="   ", evidence=[])


def test_critic_handles_empty_evidence_gracefully() -> None:
    critic = CriticAgent(llm_client=FakeGroq({}))
    result = critic.evaluate(question="Any question?", evidence=[], document_id="doc-empty")
    assert result["document_id"] == "doc-empty"
    assert result["evaluations"] == []
    assert result["contradictions"] == []
    assert result["summary"]["total_chunks"] == 0
    assert result["summary"]["has_contradictions"] is False


# ==============================================================================
# 7. FastAPI Endpoint Integration Test
# ==============================================================================
def test_critic_api_endpoint(monkeypatch) -> None:
    fake_response = {
        "evaluations": [
            {
                "chunk_id": "api-test::chunk-0001",
                "relevance": True,
                "support_status": "supported",
                "quality_assessment": {
                    "evidence_strength": "high",
                    "source_quality": "high",
                    "potential_outdated": False,
                    "details": "API test verification.",
                },
                "contradiction_status": "none",
                "explanation": "Evaluated via API endpoint successfully.",
            }
        ],
        "contradictions": [],
    }

    test_critic = CriticAgent(llm_client=FakeGroq(fake_response))
    monkeypatch.setattr(critic_api, "get_critic_agent", lambda: test_critic)

    client = TestClient(app)
    payload = {
        "document_id": "api-test",
        "question": "Does this endpoint work?",
        "evidence": [
            {
                "document_id": "api-test",
                "chunk_id": "api-test::chunk-0001",
                "filename": "test.txt",
                "page_number": 1,
                "source": "test.txt#page=1",
                "text": "This endpoint works reliably.",
            }
        ],
    }

    response = client.post("/api/critic/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == "api-test"
    assert len(data["evaluations"]) == 1
    eval_item = data["evaluations"][0]
    for key in REQUIRED_EVALUATION_KEYS:
        assert key in eval_item
    assert eval_item["relevance"] is True
    assert eval_item["support_status"] == "supported"
