"""
Standard / Normal RAG Baseline Pipeline.
Implements standard retrieve-then-generate RAG without:
- Critic Agent
- Trust Score Model
- Contradiction Detector
- Abstention Engine
- Verifier Agent
- Synthesizer Agent

Flow:
Question -> Embedding -> FAISS Retrieval -> Top-k Evidence -> LLM Generation -> Baseline Answer
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

NORMAL_RAG_SYSTEM_PROMPT = (
    "You are a standard question answering assistant. Answer the question directly and concisely "
    "based on the provided context documents. If the context does not contain the answer, "
    "do your best to answer based on what is available."
)


def run_normal_rag_generation(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    llm_client: Any | None = None,
    max_tokens: int = 300,
) -> Dict[str, Any]:
    """
    Executes standard LLM generation given question and retrieved chunks.
    No critic, no contradiction detection, no trust score, no verification, no abstention.
    """
    start_time = time.perf_counter()

    # Format context without quality judgment or filtering
    context_blocks = []
    for i, c in enumerate(retrieved_chunks):
        title = c.get("source") or c.get("title") or f"Document {i+1}"
        text = c.get("text", "").strip()
        context_blocks.append(f"[{title}]: {text}")

    context_str = "\n\n".join(context_blocks) if context_blocks else "No context available."

    if llm_client is not None and getattr(llm_client, "api_key", None):
        try:
            # Direct single LLM generation
            client = llm_client.client
            model = getattr(llm_client, "model", None) or "llama-3.3-70b-versatile"
            user_prompt = f"CONTEXT:\n{context_str}\n\nQUESTION:\n{question}\n\nANSWER:"

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": NORMAL_RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=max_tokens,
            )
            raw_answer = response.choices[0].message.content.strip()
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "answer": raw_answer,
                "latency_ms": latency_ms,
                "retrieved_chunks_count": len(retrieved_chunks),
                "is_fallback": False,
            }
        except Exception as exc:
            # Fallback if API call fails
            pass

    # Deterministic / Extractive fallback if LLM is unavailable
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    fallback_text = " ".join([c.get("text", "") for c in retrieved_chunks[:2]]).strip()
    if not fallback_text:
        fallback_text = "No relevant context found to generate an answer."

    return {
        "answer": fallback_text,
        "latency_ms": latency_ms,
        "retrieved_chunks_count": len(retrieved_chunks),
        "is_fallback": True,
    }
