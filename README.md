# Trust-Aware Multi-Agent Retrieval and Verification Framework for Hallucination-Resistant LLM Systems

## About this repository

This repository contains the implementation of a Trust-Aware Multi-Agent Retrieval and
Verification Framework for Hallucination-Resistant LLM Systems.

## Current stage: 40% baseline (Normal RAG)

At this stage, the project implements only a **baseline Retrieval-Augmented Generation (RAG)
pipeline**. This baseline lets a user upload documents, index them, and ask questions that are
answered using retrieved evidence from those documents.

The following components are **NOT yet implemented** and will be added in later stages:
- Critic Agent
- Trust Score
- Trust-Based Decision
- Synthesizer Agent
- Verifier Agent
- Contradiction Detection

## Project structure

```
trust-aware-rag/
│
├── frontend/              # Next.js/React frontend
│   ├── app/
│   ├── components/
│   ├── public/
│   └── package.json
│
├── backend/                # FastAPI backend
│   ├── app/
│   │   ├── main.py                    # FastAPI entry point
│   │   ├── api/
│   │   │   ├── upload.py              # Document upload endpoint
│   │   │   └── chat.py                # Question-answering endpoint
│   │   ├── ingestion/
│   │   │   ├── loaders.py             # Load PDF/TXT/DOC/CSV/JSON/URL content
│   │   │   ├── chunker.py             # Split text into chunks
│   │   │   └── processor.py           # Coordinate loading, cleaning, chunking, metadata
│   │   ├── embeddings/
│   │   │   └── embedding_model.py     # sentence-transformers/all-MiniLM-L6-v2
│   │   ├── retrieval/
│   │   │   ├── faiss_store.py         # Create/save/load FAISS indexes
│   │   │   └── retriever.py           # Embed query, search FAISS, return chunks
│   │   ├── llm/
│   │   │   └── groq_client.py         # Groq + Llama model for answer generation
│   │   └── database/
│   │       └── supabase.py            # Supabase Postgres for metadata
│   ├── data/uploads/                  # Uploaded documents
│   ├── indexes/faiss/                 # FAISS index files
│   ├── requirements.txt
│   ├── .env
│   └── .gitignore
│
├── README.md
└── .gitignore
```

## Document isolation

Every uploaded document is assigned a unique `document_id`. Every chunk stores:
- `document_id`
- `filename`
- `chunk_id`
- `page number` (when available)
- `source`
- `text`

Questions about a given document are answered using retrieval scoped strictly to that
document's `document_id`. Chunks from other previously uploaded documents are never mixed
into the retrieval context unless explicitly selected by the user.

## Installation

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file (already scaffolded) with:
```
GROQ_API_KEY=your_groq_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

### Frontend
```bash
cd frontend
npm install
```

## Running the project

### Run backend
```bash
cd backend
uvicorn app.main:app --reload
```

### Run frontend
```bash
cd frontend
npm run dev
```

## RAG workflow (baseline)

1. **Upload**: A document is uploaded via `api/upload.py`.
2. **Ingestion**: `ingestion/processor.py` coordinates loading (`loaders.py`), cleaning, and
   chunking (`chunker.py`) of the document, and assigns metadata (`document_id`, `chunk_id`,
   `filename`, `page`, `source`).
3. **Embedding**: Each chunk is embedded using `embeddings/embedding_model.py`
   (`sentence-transformers/all-MiniLM-L6-v2`).
4. **Indexing**: Embeddings are stored in a FAISS index via `retrieval/faiss_store.py`.
5. **Metadata storage**: Document and chunk metadata are stored in Supabase via
   `database/supabase.py`.
6. **Question answering**: When a user asks a question via `api/chat.py`:
   - The question is embedded and searched against the FAISS index for the currently selected
     `document_id` (`retrieval/retriever.py`).
   - The most relevant chunks are retrieved.
   - `llm/groq_client.py` sends the question and retrieved chunks to a Llama model on Groq to
     generate the final answer.

Later stages will extend this pipeline with a Critic Agent, Trust Score, Trust-Based Decision,
Synthesizer Agent, Verifier Agent, and Contradiction Detection to make the system
hallucination-resistant.
