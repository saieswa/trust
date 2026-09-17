"""Document-scoped semantic retrieval over the persistent FAISS store."""

from __future__ import annotations

from app.embeddings.embedding_model import EmbeddingModel, get_embedding_model
from app.database.redis_cache import RedisCache, get_cache
from app.retrieval.faiss_store import FAISSStore


class Retriever:
	"""Retrieve supporting chunks for one explicitly selected document."""

	def __init__(
		self,
		store: FAISSStore | None = None,
		embedding_model: EmbeddingModel | None = None,
		relevance_threshold: float = 0.35,
		cache: RedisCache | None = None,
	) -> None:
		if not 0 <= relevance_threshold <= 1:
			raise ValueError("relevance_threshold must be between 0 and 1")
		self.embedding_model = embedding_model or get_embedding_model()
		self.store = store or FAISSStore(embedding_model=self.embedding_model)
		self.relevance_threshold = relevance_threshold
		self.cache = cache or get_cache()

	def retrieve(
		self,
		question: str,
		document_id: str,
		top_k: int = 5,
	) -> list[dict]:
		"""Return only relevant evidence belonging to ``document_id``."""
		if not question.strip():
			raise ValueError("question must contain non-empty text")
		if not document_id.strip():
			raise ValueError("document_id must contain non-empty text")
		if top_k <= 0:
			raise ValueError("top_k must be greater than 0")

		cache_key = self.cache.query_key(document_id, question)
		cached = self.cache.get(cache_key)
		if cached is not None:
			return cached

		# Reload persisted vectors only if a new document was indexed since last query
		if hasattr(self.store, "reload_if_modified"):
			self.store.reload_if_modified()
		elif self.store.index is None:
			self.store.load()
		query_embedding = self.embedding_model.embed_query(question)
		candidates = self.store.search(
			query_embedding,
			top_k=top_k,
			document_id=document_id,
			query_text=question,
		)
		results = [
			result
			for result in candidates
			if result["document_id"] == document_id
			and result["score"] >= self.relevance_threshold
		]
		self.cache.set(cache_key, results)
		return results
