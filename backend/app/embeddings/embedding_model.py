"""Sentence-transformer embeddings for document chunks and user queries."""

from __future__ import annotations

from functools import lru_cache
import logging
from typing import Sequence

import numpy as np

from app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


class EmbeddingError(RuntimeError):
	"""Raised when the embedding model cannot load or encode input."""


@lru_cache(maxsize=1)
def _load_model(model_name: str = MODEL_NAME):
	if not model_name or not isinstance(model_name, str):
		raise EmbeddingError(f"Invalid model name: {model_name!r}")
	try:
		from sentence_transformers import SentenceTransformer

		logger.info("Initializing SentenceTransformer with model '%s'", model_name)
		return SentenceTransformer(model_name)
	except Exception as exc:
		logger.exception("Failed to load embedding model '%s'", model_name)
		raise EmbeddingError(
			f"Failed to load embedding model '{model_name}': {type(exc).__name__}: {exc}"
		) from exc


class EmbeddingModel:
	"""Shared normalized embedding model for chunks and user questions."""

	def __init__(self, model_name: str = MODEL_NAME) -> None:
		self.model_name = model_name

	@property
	def model(self):
		return _load_model(self.model_name)

	@property
	def dimension(self) -> int:
		try:
			get_dimension = getattr(
				self.model,
				"get_embedding_dimension",
				getattr(self.model, "get_sentence_embedding_dimension", None),
			)
			dim = int(get_dimension()) if callable(get_dimension) else EMBEDDING_DIMENSION
			if dim != EMBEDDING_DIMENSION:
				raise EmbeddingError(
					f"Loaded model dimension {dim} does not match expected dimension {EMBEDDING_DIMENSION}"
				)
			return dim
		except EmbeddingError:
			raise
		except Exception as exc:
			logger.exception("Failed to determine embedding dimension")
			raise EmbeddingError(
				f"Failed to determine embedding dimension: {type(exc).__name__}: {exc}"
			) from exc

	def _encode(self, texts: Sequence[str]) -> np.ndarray:
		if not texts:
			return np.empty((0, self.dimension), dtype=np.float32)

		safe_texts: list[str] = []
		for idx, text in enumerate(texts):
			if text is None:
				raise ValueError(f"Embedding input text at index {idx} cannot be None.")
			val = str(text).strip()
			if not val:
				raise ValueError(f"Embedding input text at index {idx} must contain non-empty text.")
			safe_texts.append(val)

		try:
			vectors = self.model.encode(
				safe_texts,
				convert_to_numpy=True,
				normalize_embeddings=True,
			)
		except Exception as exc:
			logger.exception("Failed to generate embedding for %d chunks", len(safe_texts))
			raise EmbeddingError(
				f"Failed to generate embedding for {len(safe_texts)} chunks: {type(exc).__name__}: {exc}"
			) from exc

		result = np.asarray(vectors, dtype=np.float32)
		if result.ndim == 1:
			result = result.reshape(1, -1)

		if result.shape[1] != self.dimension:
			raise EmbeddingError(
				f"Generated embedding dimension {result.shape[1]} does not match expected {self.dimension}"
			)
		if np.isnan(result).any():
			raise EmbeddingError("Generated embeddings contain NaN values.")
		if np.isinf(result).any():
			raise EmbeddingError("Generated embeddings contain infinite values.")

		return result

	def embed_chunks(self, chunks: Sequence[Chunk]) -> np.ndarray:
		"""Return one normalized vector per document chunk."""
		if not chunks:
			return np.empty((0, self.dimension), dtype=np.float32)
		return self._encode([chunk.text for chunk in chunks])

	@lru_cache(maxsize=1024)
	def _cached_embed_query(self, query: str) -> bytes:
		vectors = self._encode([query])
		return vectors[0].tobytes()

	def embed_query(self, question: str) -> np.ndarray:
		"""Return one normalized vector for a user question with in-memory caching."""
		if question is None:
			raise ValueError("Embedding query question cannot be None.")
		val = str(question).strip()
		if not val:
			raise ValueError("Embedding query question must contain non-empty text.")
		raw = self._cached_embed_query(val)
		return np.frombuffer(raw, dtype=np.float32)


_embedding_model = EmbeddingModel()


def get_embedding_model() -> EmbeddingModel:
	"""Return the process-wide embedding component."""
	return _embedding_model
