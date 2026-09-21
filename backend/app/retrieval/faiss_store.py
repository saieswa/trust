"""Persistent FAISS vector storage with document-scoped retrieval."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import Sequence

import numpy as np

from app.embeddings.embedding_model import EmbeddingModel, get_embedding_model
from app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

INDEX_DIR = Path(__file__).resolve().parents[2] / "indexes" / "faiss"
INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.json"
EMBEDDING_DIMENSION = 384


class FAISSStoreError(RuntimeError):
	"""Raised when a FAISS store cannot be created, loaded, or searched."""


class FAISSStore:
	"""Store normalized chunk vectors and searchable chunk metadata on disk."""

	def __init__(
		self,
		index_dir: str | Path = INDEX_DIR,
		embedding_model: EmbeddingModel | None = None,
	) -> None:
		self.index_dir = Path(index_dir)
		# Case D: ensure directory exists automatically
		self.index_dir.mkdir(parents=True, exist_ok=True)
		self.index_path = self.index_dir / INDEX_FILENAME
		self.metadata_path = self.index_dir / METADATA_FILENAME
		self.embedding_model = embedding_model or get_embedding_model()
		self._index = None
		self._metadata: list[dict] = []
		self.load()

	@property
	def index(self):
		return self._index

	@property
	def metadata(self) -> list[dict]:
		return list(self._metadata)

	@property
	def size(self) -> int:
		return 0 if self._index is None else self._index.ntotal

	@property
	def dimension(self) -> int:
		return self.embedding_model.dimension

	def _new_index(self):
		try:
			import faiss

			return faiss.IndexFlatIP(self.dimension)
		except Exception as exc:
			logger.exception("Failed to create FAISS index")
			raise FAISSStoreError(
				f"Failed to create FAISS index: {type(exc).__name__}: {exc}"
			) from exc

	def load(self) -> None:
		"""Load the persisted index and metadata, or initialize an empty store."""
		# Case A: Neither index nor metadata exist -> create empty index
		if not self.index_path.exists() and not self.metadata_path.exists():
			self._index = self._new_index()
			self._metadata = []
			return

		# Case C: Corrupted or incomplete store
		if not self.index_path.exists() or not self.metadata_path.exists():
			raise FAISSStoreError(
				"FAISS store corrupted: index.faiss and metadata.json must be present together."
			)

		# Case B: Both exist -> load and validate
		try:
			import faiss

			self._index = faiss.read_index(str(self.index_path))
			self._metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
			if not isinstance(self._metadata, list):
				raise ValueError("metadata must be a list")
			if self._index.ntotal != len(self._metadata):
				raise ValueError(
					f"index and metadata counts differ: index={self._index.ntotal}, metadata={len(self._metadata)}"
				)
			if self._index.d != self.dimension:
				raise ValueError(
					f"index dimension ({self._index.d}) differs from embedding model ({self.dimension})"
				)
			try:
				self._last_loaded_mtime = max(
					self.index_path.stat().st_mtime,
					self.metadata_path.stat().st_mtime,
				)
			except Exception:
				self._last_loaded_mtime = 0.0
		except FAISSStoreError:
			raise
		except Exception as exc:
			logger.exception("Failed to load FAISS store")
			raise FAISSStoreError(
				f"Failed to load FAISS store: {type(exc).__name__}: {exc}"
			) from exc

	def reload_if_modified(self) -> bool:
		"""Reload index and metadata only if on-disk files are newer than in-memory copy."""
		if self._index is None:
			self.load()
			return True
		if not self.index_path.exists() or not self.metadata_path.exists():
			return False
		try:
			current_mtime = max(
				self.index_path.stat().st_mtime,
				self.metadata_path.stat().st_mtime,
			)
			if current_mtime > getattr(self, "_last_loaded_mtime", 0.0):
				self.load()
				return True
		except Exception:
			pass
		return False

	def _save(self) -> None:
		if self._index is None:
			return
		try:
			import faiss

			self.index_dir.mkdir(parents=True, exist_ok=True)
			faiss.write_index(self._index, str(self.index_path))
			self.metadata_path.write_text(
				json.dumps(self._metadata, ensure_ascii=False, indent=2),
				encoding="utf-8",
			)
		except Exception as exc:
			logger.exception("Failed to save FAISS store")
			raise FAISSStoreError(
				f"Failed to save FAISS store: {type(exc).__name__}: {exc}"
			) from exc

	def add_chunks(
		self,
		chunks: Sequence[Chunk],
		embeddings: np.ndarray | None = None,
	) -> int:
		"""Append new chunks and metadata without replacing existing vectors."""
		if not chunks:
			return 0

		existing_ids = {record["chunk_id"] for record in self._metadata}
		new_chunks = [chunk for chunk in chunks if chunk.chunk_id not in existing_ids]
		if not new_chunks:
			return 0

		if embeddings is None:
			vectors = self.embedding_model.embed_chunks(new_chunks)
		else:
			vectors = np.asarray(embeddings, dtype=np.float32)
			if vectors.shape[0] != len(chunks):
				raise ValueError("embeddings must contain one vector per chunk")
			vectors = vectors[[chunks.index(chunk) for chunk in new_chunks]]

		if vectors.ndim != 2 or vectors.shape[0] != len(new_chunks):
			raise ValueError("embeddings must be a two-dimensional array")
		if vectors.shape[1] != self.dimension:
			raise ValueError(
				f"embedding dimension ({vectors.shape[1]}) does not match expected ({self.dimension})"
			)

		norms = np.linalg.norm(vectors, axis=1, keepdims=True)
		if np.any(norms == 0):
			raise ValueError("embeddings must contain non-zero vectors")
		vectors = vectors / norms

		if self._index is None:
			self._index = self._new_index()
		self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))
		self._metadata.extend(
			{
				"document_id": chunk.document_id,
				"chunk_id": chunk.chunk_id,
				"filename": chunk.filename,
				"page_number": chunk.page_number,
				"text": chunk.text,
				"source": chunk.source,
			}
			for chunk in new_chunks
		)
		self._save()
		return len(new_chunks)

	def add_chunks_batched(
		self,
		chunks: Sequence[Chunk],
		batch_size: int = 64,
	):
		"""Embed and index chunks in batches, persisting after every batch.

		This keeps RAM usage bounded for large documents (300+ pages) and
		ensures partial progress is never lost if an error occurs mid-way.

		Args:
			chunks: All chunks to embed and index.
			batch_size: Number of chunks per embedding + FAISS add call (default 64).

		Yields:
			Tuple[int, int]: (chunks_done_so_far, total_chunks) after each batch.
		"""
		if not chunks:
			return

		existing_ids = {record["chunk_id"] for record in self._metadata}
		new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
		total = len(new_chunks)
		if total == 0:
			return

		batch_size = max(1, batch_size)
		done = 0

		if self._index is None:
			self._index = self._new_index()

		for start in range(0, total, batch_size):
			batch = new_chunks[start : start + batch_size]
			vectors = self.embedding_model.embed_chunks(batch)

			if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
				raise FAISSStoreError(
					f"Embedding dimension mismatch in batch starting at {start}"
				)

			norms = np.linalg.norm(vectors, axis=1, keepdims=True)
			zero_mask = norms.flatten() == 0
			if zero_mask.any():
				# Replace zero-norm vectors with unit vectors to avoid divide-by-zero
				vectors[zero_mask] = np.zeros(self.dimension, dtype=np.float32)
				vectors[zero_mask, 0] = 1.0
				norms = np.where(norms == 0, 1.0, norms)
			vectors = vectors / norms

			self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))
			self._metadata.extend(
				{
					"document_id": chunk.document_id,
					"chunk_id": chunk.chunk_id,
					"filename": chunk.filename,
					"page_number": chunk.page_number,
					"text": chunk.text,
					"source": chunk.source,
				}
				for chunk in batch
			)
			# Persist after every batch so progress survives errors
			self._save()
			done += len(batch)
			logger.info("[FAISS] batched add: %d/%d chunks indexed", done, total)
			yield done, total



	def search(
		self,
		query: str | np.ndarray,
		top_k: int = 5,
		document_id: str | None = None,
		query_text: str | None = None,
	) -> list[dict]:
		"""Return highest-scoring metadata records, optionally document-scoped."""
		if top_k <= 0:
			raise ValueError("top_k must be greater than 0")
		if self._index is None or self._index.ntotal == 0:
			return []

		query_str = query if isinstance(query, str) else query_text
		query_vector = (
			self.embedding_model.embed_query(query)
			if isinstance(query, str)
			else np.asarray(query, dtype=np.float32)
		)
		query_vector = query_vector.reshape(1, -1)
		norm = np.linalg.norm(query_vector)
		if norm == 0:
			raise ValueError("query embedding must be non-zero")
		query_vector = query_vector / norm

		scores, positions = self._index.search(query_vector, self._index.ntotal)
		candidates: list[dict] = []
		for score, position in zip(scores[0], positions[0]):
			pos = int(position)
			if pos < 0 or pos >= len(self._metadata):
				continue
			record = self._metadata[pos]
			if document_id is not None and record.get("document_id") != document_id:
				continue
			dense_score = float(score)
			final_score = compute_hybrid_score(query_str, record.get("text", ""), dense_score)
			candidates.append({**record, "score": final_score, "dense_score": dense_score})
			if len(candidates) >= max(top_k * 5, 50):
				break

		# Re-rank candidates by final score descending
		candidates.sort(key=lambda r: r["score"], reverse=True)
		return candidates[:top_k]


def compute_hybrid_score(query: str | None, text: str, dense_score: float) -> float:
	"""Compute hybrid relevance score combining dense similarity and lexical match.

	- Exact phrase match: User query appears verbatim (e.g. section titles or headings).
	- Content word overlap: Matches on non-stopwords with full or partial overlap.
	- Definitional boost: Questions asking 'what is/are', 'explain', 'define' prioritize
	  core explanatory sentences ('X is', 'X refers to', 'Core Idea', 'Overview') over
	  troubleshooting or failure mode sections ('What Breaks It').
	- Irrelevant text: Preserves dense score below relevance threshold (no false positives).
	"""
	if not query or not isinstance(query, str) or not query.strip():
		return float(dense_score)
	if not text or not isinstance(text, str) or not text.strip():
		return float(dense_score)

	norm_q = " ".join(re.sub(r"[^\w\s]", " ", query.lower()).split())
	norm_t = " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())
	if not norm_q or not norm_t:
		return float(dense_score)

	d_score = float(dense_score)
	score = d_score

	# 1. Exact phrase match: The normalized query appears verbatim in the chunk text
	if len(norm_q) >= 3 and norm_q in norm_t:
		phrase_score = 0.75 + 0.20 * max(0.0, d_score)
		score = max(score, phrase_score)

	# 2. Key content word overlap (excluding common English stopwords)
	q_words = [w for w in re.findall(r"\b\w+\b", norm_q) if len(w) > 1]
	stop_words = {
		"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
		"in", "on", "at", "to", "for", "with", "by", "about", "against", "between",
		"into", "through", "during", "before", "after", "above", "below", "from",
		"up", "down", "out", "off", "over", "under", "again", "further",
		"then", "once", "here", "there", "when", "where", "why", "how", "all",
		"any", "both", "each", "few", "more", "most", "other", "some", "such",
		"no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
		"s", "t", "can", "will", "just", "don", "should", "now", "it", "its",
		"do", "does", "did", "what", "which",
	}
	content_words = [w for w in q_words if w not in stop_words]
	if not content_words:
		content_words = q_words

	if content_words:
		matches = sum(1 for w in content_words if re.search(r"\b" + re.escape(w) + r"\b", norm_t))
		ratio = matches / len(content_words)
		if ratio == 1.0 and len(content_words) >= 2:
			overlap_score = 0.65 + 0.20 * max(0.0, d_score)
			score = max(score, overlap_score)
		elif ratio >= 0.5 and len(content_words) >= 2:
			combined = 0.50 * d_score + 0.35 * ratio
			score = max(score, combined)

		# 3. Definitional query boost
		is_definitional_query = bool(
			re.search(r"\b(what\s+is|what\s+are|define|explain|meaning\s+of|overview\s+of)\b", norm_q)
		)
		if is_definitional_query and len(content_words) >= 1:
			core_term = " ".join(content_words)
			# Look for definitional constructs in text: 'term is', 'term are', 'core idea', 'definition'
			has_definitional_phrase = bool(
				re.search(r"\b" + re.escape(core_term) + r"\s+(is|are|refers\s+to|means|fundamentally)\b", norm_t)
			)
			has_intro_heading = bool(
				re.search(r"\b(core\s+idea|overview|definition|first\s+principles|introduction)\b", norm_t)
			)
			if has_definitional_phrase or has_intro_heading:
				score += 0.15

			# Penalize troubleshooting/failure chunks when asking for definition
			is_troubleshooting_chunk = bool(
				re.search(r"\b(what\s+breaks\s+it|edge\s+cases|pitfalls|common\s+mistakes|disadvantages|limitations)\b", norm_t)
			)
			if is_troubleshooting_chunk and not re.search(r"\b(break|fail|edge|limit|disadvantage|mistake)\b", norm_q):
				score -= 0.12

	return round(min(1.0, max(0.0, score)), 4)



