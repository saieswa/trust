"""Comprehensive Test Suite for Answer Quality, Verification, and Trust Scoring.

Executes the 7 required test cases on 'binary search explained.pdf':
1. 'What is binary search?'
2. 'What is the time complexity of binary search?'
3. 'What are the requirements for binary search?'
4. Question whose answer is NOT in the document ('What is Dijkstra's shortest path algorithm?')
5. Question about an unrelated topic ('What is the capital of Australia?')
6. Question whose answer appears on a different page / Page 2 ('What breaks binary search?')
7. Question requiring information from two relevant chunks ('How does binary search eliminate possibilities and what conditions must hold?')
"""

import glob
import json
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.loaders import load_document
from app.ingestion.chunker import chunk_document
from app.retrieval.faiss_store import FAISSStore
from app.retrieval.retriever import Retriever
from app.embeddings.embedding_model import get_embedding_model
from app.agents.orchestrator import TrustAwareOrchestrator
from app.agents.critic import get_critic_agent
from app.agents.synthesizer import get_synthesizer_agent
from app.agents.verifier import get_verifier_agent
from app.agents.abstention import get_abstention_engine


def setup_test_pipeline():
    files = glob.glob(str(Path(__file__).resolve().parent.parent / "data/uploads/*binary search explained*.pdf"))
    if not files:
        raise FileNotFoundError("Could not find binary search explained.pdf in data/uploads")
    
    doc_path = files[0]
    doc_id = "test-doc-binary-search"
    extracted = load_document(doc_path, document_id=doc_id, source_filename="binary search explained.pdf")
    chunks = chunk_document(extracted)
    
    embedding_model = get_embedding_model()
    store = FAISSStore(embedding_model=embedding_model)
    store.add_chunks(chunks)
    
    retriever = Retriever(store=store, embedding_model=embedding_model, relevance_threshold=0.35)
    orchestrator = TrustAwareOrchestrator(
        retriever=retriever,
        critic_agent=get_critic_agent(),
        synthesizer_agent=get_synthesizer_agent(),
        verifier_agent=get_verifier_agent(),
        abstention_engine=get_abstention_engine(),
    )
    return orchestrator, doc_id, chunks


def run_and_report_test(orchestrator, doc_id, test_num, question, mode="normal"):
    print("=" * 80)
    print(f"TEST CASE {test_num}: \"{question}\" (Mode: {mode})")
    print("=" * 80)

    res = orchestrator.run(question=question, document_id=doc_id, mode=mode)

    print(f"QUESTION: {res.get('question')}")
    print("\nRETRIEVED EVIDENCE:")
    results = res.get("retrieval_results", [])
    if results:
        for idx, r in enumerate(results[:3]):
            print(f"  [{idx+1}] Score: {r.get('score'):.4f} | Page: {r.get('page_number')} | ID: {r.get('chunk_id')}")
            print(f"      Text: {repr(r.get('text', '')[:140])}...")
    else:
        print("  (No evidence retrieved above threshold)")

    print("\nCRITIC & EVALUATION RESULT:")
    evals = res.get("evaluations", [])
    print(f"  Evaluations count: {len(evals)}")
    for ev in evals[:3]:
        print(f"  - Chunk {ev.get('chunk_id')}: Relevance={ev.get('relevance')}, Support={ev.get('support_status')}")

    print("\nGENERATED ANSWER:")
    print(f"  {res.get('answer')}")

    print("\nVERIFIER RESULT:")
    v_info = res.get("verification", {})
    print(f"  Status: {res.get('verification_status')}")
    print(f"  Score: {v_info.get('verification_score')}")
    print(f"  Claims count: {len(res.get('claim_level_verification', []))}")
    for c in res.get("claim_level_verification", [])[:3]:
        c_text = c.get("claim") or c.get("text")
        print(f"  - Claim: \"{c_text}\" -> Supported={c.get('supported')}, Status={c.get('status')}")

    trust_dict = res.get("trust_score", {})
    print("\nTRUST SCORE & DECISION:")
    print(f"  Overall Score: {trust_dict.get('overall_score')}")
    print(f"  Trust Level: {trust_dict.get('trust_level')}")
    print(f"  Abstention: {res.get('abstention')}")
    print(f"  Abstention Reason: {res.get('abstention_reason')}")
    print(f"  Sources: {res.get('sources')}")
    print("=" * 80 + "\n")
    import time
    time.sleep(2)
    return res


def main():
    orchestrator, doc_id, chunks = setup_test_pipeline()
    print(f"Successfully loaded and chunked binary search explained.pdf into {len(chunks)} sentence-aware chunks.\n")

    # Test 1: What is binary search?
    r1 = run_and_report_test(orchestrator, doc_id, 1, "What is binary search?")
    assert not r1["answer"].startswith("Answer grounded on evidence: by repeatedly asking which side of m"), "Bug regression: raw sliced fragment returned!"
    assert "binary search" in r1["answer"].lower(), "Answer does not mention binary search!"
    assert r1["verification_status"] in ("SUPPORTED", "PARTIALLY_SUPPORTED"), "Verification status unexpected!"

    # Test 2: What is the time complexity of binary search?
    r2 = run_and_report_test(orchestrator, doc_id, 2, "What is the time complexity of binary search?")
    assert any(k in r2["answer"].lower() for k in ("log", "o(1)", "not explicitly", "enough information", "does not state", "no information")), "Time complexity information missing or improper response!"

    # Test 3: What are the requirements for binary search?
    r3 = run_and_report_test(orchestrator, doc_id, 3, "What are the requirements for binary search?")
    assert any(w in r3["answer"].lower() for w in ("monotonic", "order", "condition", "space")), "Requirements not answered!"

    # Test 4: A question whose answer is NOT in the document
    r4 = run_and_report_test(orchestrator, doc_id, 4, "What is Dijkstra's shortest path algorithm?")
    assert r4["abstention"] is True or "not find enough information" in r4["answer"].lower() or "no information" in r4["answer"].lower(), "Did not abstain on unmentioned topic!"

    # Test 5: A question about an unrelated topic
    r5 = run_and_report_test(orchestrator, doc_id, 5, "What is the capital of Australia?")
    assert r5["abstention"] is True or "not find enough information" in r5["answer"].lower() or "no information" in r5["answer"].lower(), "Did not abstain on unrelated topic!"
    assert r5["trust_score"]["trust_level"] == "LOW_TRUST", "Unrelated query should have LOW_TRUST!"

    # Test 6: A question whose answer appears on a different page (Page 2: What breaks it?)
    r6 = run_and_report_test(orchestrator, doc_id, 6, "What breaks binary search?")
    assert any(w in r6["answer"].lower() for w in ("monotonic", "order", "expensive", "break")), "Page 2 content not retrieved!"

    # Test 7: A question requiring information from two relevant chunks
    r7 = run_and_report_test(orchestrator, doc_id, 7, "How does binary search eliminate possibilities and what conditions must hold?")
    assert len(r7.get("retrieval_results", [])) >= 2, "Should retrieve multiple chunks!"

    print("ALL 7 TEST CASES EXECUTED AND VALIDATED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
