"""Tests for optional document-scoped Redis caching."""

from __future__ import annotations

from app.database.redis_cache import RedisCache


class FakeRedis:
    def __init__(self):
        self.values = {}

    def get(self, key):
        return self.values.get(key)

    def setex(self, key, _ttl, value):
        self.values[key] = value


def test_cache_keys_isolate_two_documents() -> None:
    cache = RedisCache(client=FakeRedis())
    paper_a_key = cache.query_key("paper-a", "What is attention?")
    paper_b_key = cache.query_key("paper-b", "What is attention?")

    assert paper_a_key.startswith("rag:paper-a:")
    assert paper_b_key.startswith("rag:paper-b:")
    assert paper_a_key != paper_b_key

    cache.set(paper_a_key, [{"document_id": "paper-a"}])
    assert cache.get(paper_a_key) == [{"document_id": "paper-a"}]
    assert cache.get(paper_b_key) is None


def test_cache_is_disabled_without_credentials() -> None:
    cache = RedisCache(url="", token="")
    assert not cache.enabled
    assert cache.get("rag:paper-a:query") is None
    assert not cache.set("rag:paper-a:query", {"document_id": "paper-a"})