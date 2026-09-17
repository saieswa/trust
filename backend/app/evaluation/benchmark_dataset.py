"""Curated benchmark dataset for empirical evaluation of Baseline RAG vs. Trust-Aware Multi-Agent RAG.

Spans 5 stress categories:
1. Answerable from documents
2. Not answerable from documents
3. Ambiguous questions
4. Questions involving conflicting evidence
5. Questions designed to test hallucination
"""

from __future__ import annotations

from typing import Any

BENCHMARK_CASES: list[dict[str, Any]] = [
    # =========================================================================
    # Category 1: Questions Answerable from Documents
    # =========================================================================
    {
        "id": "case-01-answerable-transformer",
        "category": "answerable",
        "category_label": "Answerable from Document",
        "question": "What is the Transformer architecture based on?",
        "document_id": "doc-attention-paper",
        "evidence": [
            {
                "chunk_id": "doc-attention-paper::c1",
                "document_id": "doc-attention-paper",
                "filename": "attention_is_all_you_need.pdf",
                "page_number": 1,
                "source": "attention_is_all_you_need.pdf#page=1",
                "text": "The Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution.",
                "score": 0.96,
            }
        ],
        "expected_action": "DIRECT_ANSWER",
        "expected_trust_level": "HIGH",
        "expect_abstention": False,
        "ground_truth_answer": "The Transformer relies entirely on self-attention mechanisms without using sequence-aligned RNNs or convolutions.",
        "key_phrases": ["self-attention", "attention"],
        "expect_contradiction": False,
        "ground_truth_faithfulness": 1.0,
    },
    {
        "id": "case-02-answerable-binary-search",
        "category": "answerable",
        "category_label": "Answerable from Document",
        "question": "What is the worst-case time complexity of binary search?",
        "document_id": "doc-algorithms-handbook",
        "evidence": [
            {
                "chunk_id": "doc-algorithms-handbook::c1",
                "document_id": "doc-algorithms-handbook",
                "filename": "algorithms_handbook.pdf",
                "page_number": 14,
                "source": "algorithms_handbook.pdf#page=14",
                "text": "Binary search halves the search space at each iteration, yielding an optimal worst-case time complexity of O(log n) comparisons on sorted arrays.",
                "score": 0.94,
            }
        ],
        "expected_action": "DIRECT_ANSWER",
        "expected_trust_level": "HIGH",
        "expect_abstention": False,
        "ground_truth_answer": "The worst-case time complexity of binary search is O(log n).",
        "key_phrases": ["O(log n)", "log n"],
        "expect_contradiction": False,
        "ground_truth_faithfulness": 1.0,
    },

    # =========================================================================
    # Category 2: Questions Not Answerable from Documents
    # =========================================================================
    {
        "id": "case-03-unanswerable-coca-cola",
        "category": "not_answerable",
        "category_label": "Not Answerable from Document",
        "question": "What is the secret recipe for Coca-Cola according to this paper?",
        "document_id": "doc-attention-paper",
        "evidence": [
            {
                "chunk_id": "doc-attention-paper::c2",
                "document_id": "doc-attention-paper",
                "filename": "attention_is_all_you_need.pdf",
                "page_number": 5,
                "source": "attention_is_all_you_need.pdf#page=5",
                "text": "Table 1 shows the BLEU scores achieved by the Transformer on the WMT 2014 English-to-German translation task.",
                "score": 0.18,
            }
        ],
        "expected_action": "ABSTAIN",
        "expected_trust_level": "LOW",
        "expect_abstention": True,
        "abstention_reason": "insufficient_evidence",
        "ground_truth_answer": "I don't have enough reliable evidence in the uploaded document to answer this question.",
        "expect_contradiction": False,
        "ground_truth_faithfulness": 0.0,
    },
    {
        "id": "case-04-unanswerable-stock-price",
        "category": "not_answerable",
        "category_label": "Not Answerable from Document",
        "question": "What was the Tesla stock price on July 14, 2024?",
        "document_id": "doc-algorithms-handbook",
        "evidence": [
            {
                "chunk_id": "doc-algorithms-handbook::c2",
                "document_id": "doc-algorithms-handbook",
                "filename": "algorithms_handbook.pdf",
                "page_number": 22,
                "source": "algorithms_handbook.pdf#page=22",
                "text": "Quicksort has an average-case running time of O(n log n) and sorts in place using partitioned recursive sub-arrays.",
                "score": 0.15,
            }
        ],
        "expected_action": "ABSTAIN",
        "expected_trust_level": "LOW",
        "expect_abstention": True,
        "abstention_reason": "insufficient_evidence",
        "ground_truth_answer": "I don't have enough reliable evidence in the uploaded document to answer this question.",
        "expect_contradiction": False,
        "ground_truth_faithfulness": 0.0,
    },

    # =========================================================================
    # Category 3: Ambiguous Questions
    # =========================================================================
    {
        "id": "case-05-ambiguous-evaluation",
        "category": "ambiguous",
        "category_label": "Ambiguous / Incomplete Evidence",
        "question": "How well did the system perform on benchmarks?",
        "document_id": "doc-model-eval",
        "evidence": [
            {
                "chunk_id": "doc-model-eval::c1",
                "document_id": "doc-model-eval",
                "filename": "evaluation_summary.pdf",
                "page_number": 3,
                "source": "evaluation_summary.pdf#page=3",
                "text": "Preliminary test results vary across tasks. English-to-German attained 28.4 BLEU, but latency was not formally recorded.",
                "score": 0.65,
            }
        ],
        "expected_action": "RETRIEVE_MORE",
        "expected_trust_level": "MEDIUM",
        "expect_abstention": False,
        "ground_truth_answer": "Performance varies across tasks (e.g. 28.4 BLEU on English-to-German), but comprehensive benchmark performance remains incomplete.",
        "key_phrases": ["BLEU", "preliminary"],
        "expect_contradiction": False,
        "ground_truth_faithfulness": 0.8,
    },
    {
        "id": "case-06-ambiguous-release-date",
        "category": "ambiguous",
        "category_label": "Ambiguous / Incomplete Evidence",
        "question": "When is the official release date for the next version?",
        "document_id": "doc-release-notes",
        "evidence": [
            {
                "chunk_id": "doc-release-notes::c1",
                "document_id": "doc-release-notes",
                "filename": "roadmap.pdf",
                "page_number": 2,
                "source": "roadmap.pdf#page=2",
                "text": "Deployment is targeted tentatively for either Q3 or Q4 depending on testing results.",
                "score": 0.62,
            }
        ],
        "expected_action": "RETRIEVE_MORE",
        "expected_trust_level": "MEDIUM",
        "expect_abstention": False,
        "ground_truth_answer": "The official release date is tentative, targeting either Q3 or Q4 subject to validation.",
        "key_phrases": ["Q3", "Q4", "tentative"],
        "expect_contradiction": False,
        "ground_truth_faithfulness": 0.8,
    },

    # =========================================================================
    # Category 4: Questions Involving Conflicting Evidence
    # =========================================================================
    {
        "id": "case-07-conflicting-accuracy",
        "category": "conflicting",
        "category_label": "Conflicting Evidence",
        "question": "What is the accuracy of the model on the validation dataset?",
        "document_id": "doc-conflicting-benchmarks",
        "evidence": [
            {
                "chunk_id": "doc-conflicting-benchmarks::c1",
                "document_id": "doc-conflicting-benchmarks",
                "filename": "report_a.pdf",
                "page_number": 4,
                "source": "report_a.pdf#page=4",
                "text": "The model accuracy is 85% on the ImageNet validation dataset after 100 epochs.",
                "score": 0.89,
            },
            {
                "chunk_id": "doc-conflicting-benchmarks::c2",
                "document_id": "doc-conflicting-benchmarks",
                "filename": "report_b.pdf",
                "page_number": 8,
                "source": "report_b.pdf#page=8",
                "text": "The model accuracy is 72% on the ImageNet validation dataset after 100 epochs.",
                "score": 0.88,
            },
        ],
        "expected_action": "ABSTAIN",
        "expected_trust_level": "LOW",
        "expect_abstention": True,
        "abstention_reason": "contradiction_detected",
        "ground_truth_answer": "I cannot answer this question because conflicting factual statements were detected within the document passages (85% vs 72%).",
        "expect_contradiction": True,
        "ground_truth_faithfulness": 0.0,
    },
    {
        "id": "case-08-conflicting-launch-date",
        "category": "conflicting",
        "category_label": "Conflicting Evidence",
        "question": "When was Project Apollo launched?",
        "document_id": "doc-apollo-history",
        "evidence": [
            {
                "chunk_id": "doc-apollo-history::c1",
                "document_id": "doc-apollo-history",
                "filename": "press_release.pdf",
                "page_number": 1,
                "source": "press_release.pdf#page=1",
                "text": "Project Apollo was officially launched in March 2021 across European operations.",
                "score": 0.87,
            },
            {
                "chunk_id": "doc-apollo-history::c2",
                "document_id": "doc-apollo-history",
                "filename": "annual_review.pdf",
                "page_number": 11,
                "source": "annual_review.pdf#page=11",
                "text": "Project Apollo launch was delayed until November 2023 due to regulatory compliance.",
                "score": 0.85,
            },
        ],
        "expected_action": "ABSTAIN",
        "expected_trust_level": "LOW",
        "expect_abstention": True,
        "abstention_reason": "contradiction_detected",
        "ground_truth_answer": "Conflicting launch dates were reported in the document (March 2021 vs November 2023).",
        "expect_contradiction": True,
        "ground_truth_faithfulness": 0.0,
    },

    # =========================================================================
    # Category 5: Questions Designed to Test Hallucination
    # =========================================================================
    {
        "id": "case-09-hallucination-false-premise-lstm",
        "category": "hallucination_testing",
        "category_label": "Hallucination Testing (False Premise)",
        "question": "Why did the Transformer architecture incorporate recurrent LSTM units in its encoder?",
        "document_id": "doc-attention-paper",
        "evidence": [
            {
                "chunk_id": "doc-attention-paper::c1",
                "document_id": "doc-attention-paper",
                "filename": "attention_is_all_you_need.pdf",
                "page_number": 1,
                "source": "attention_is_all_you_need.pdf#page=1",
                "text": "The Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution.",
                "score": 0.90,
            }
        ],
        "expected_action": "DIRECT_ANSWER",
        "expected_trust_level": "HIGH",
        "expect_abstention": False,
        "ground_truth_answer": "The Transformer does not incorporate recurrent LSTM units; it relies entirely on self-attention without recurrence.",
        "key_phrases": ["does not", "without", "entirely"],
        "expect_contradiction": False,
        "ground_truth_faithfulness": 1.0,
    },
    {
        "id": "case-10-hallucination-invented-author",
        "category": "hallucination_testing",
        "category_label": "Hallucination Testing (False Premise)",
        "question": "What role did Geoffrey Hinton play in authoring the Transformer in this paper?",
        "document_id": "doc-attention-paper",
        "evidence": [
            {
                "chunk_id": "doc-attention-paper::c3",
                "document_id": "doc-attention-paper",
                "filename": "attention_is_all_you_need.pdf",
                "page_number": 1,
                "source": "attention_is_all_you_need.pdf#page=1",
                "text": "Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin.",
                "score": 0.85,
            }
        ],
        "expected_action": "DIRECT_ANSWER",
        "expected_trust_level": "HIGH",
        "expect_abstention": False,
        "ground_truth_answer": "Geoffrey Hinton is not listed as an author of this paper (authored by Vaswani et al.).",
        "key_phrases": ["not", "Vaswani"],
        "expect_contradiction": False,
        "ground_truth_faithfulness": 1.0,
    },
]
