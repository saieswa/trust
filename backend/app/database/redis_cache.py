"""Optional Upstash Redis caching for document-scoped RAG operations."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from app.config import settings


class RedisCache:
    """Small fail-open cache wrapper; RAG remains correct when Redis is absent."""

    def __init__(
        self,
        client: Any | None = None,
        url: str | None = None,
        token: str | None = None,
        ttl_seconds: int = 300,
    ) -> None:
        self._client = client
        self.url = url if url is not None else settings.UPSTASH_REDIS_REST_URL
        self.token = token if token is not None else settings.UPSTASH_REDIS_REST_TOKEN
        self.ttl_seconds = ttl_seconds

    @property
    def enabled(self) -> bool:
        return self._client is not None or bool(self.url and self.token)

    @property
    def client(self) -> Any | None:
        if self._client is None and self.url and self.token:
            try:
                from upstash_redis import Redis

                self._client = Redis(url=self.url, token=self.token)
            except Exception:
                return None
        return self._client

    @staticmethod
    def query_key(document_id: str, question: str) -> str:
        """Build a document-scoped key for a normalized question."""
        query_hash = hashlib.sha256(" ".join(question.split()).lower().encode()).hexdigest()
        return f"rag:{document_id}:{query_hash}"

    def get(self, key: str) -> Any | None:
        client = self.client
        if client is None:
            return None
        try:
            value = client.get(key)
            return json.loads(value) if isinstance(value, str) else value
        except Exception:
            return None

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> bool:
        client = self.client
        if client is None:
            return False
        try:
            payload = json.dumps(value, ensure_ascii=False)
            ttl = self.ttl_seconds if ttl_seconds is None else ttl_seconds
            if hasattr(client, "setex"):
                client.setex(key, ttl, payload)
            else:
                client.set(key, payload, ex=ttl)
            return True
        except Exception:
            return False


_cache = RedisCache()


def get_cache() -> RedisCache:
    """Return the process-wide optional cache."""
    return _cache