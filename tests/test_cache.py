#tests/test_cache.py
import pytest
from stagerag.cache import LRUCache

def test_cache_get_set():
    cache = LRUCache(max_size=2)
    cache.set("model1", "prompt1", "step1", "result1")
    result = cache.get("model1", "prompt1", "step1")
    assert result == "result1"

def test_cache_eviction():
    cache = LRUCache(max_size=2)
    cache.set("m1", "p1", "s1", "r1")
    cache.set("m2", "p2", "s2", "r2")
    cache.set("m3", "p3", "s3", "r3")  # Should evict first
    assert cache.get("m1", "p1", "s1") is None
    assert cache.get("m3", "p3", "s3") == "r3"

def test_cache_stats():
    cache = LRUCache(max_size=10)
    cache.set("m", "p", "s", "r")
    cache.get("m", "p", "s")  # hit
    cache.get("m", "x", "s")  # miss
    stats = cache.stats()
    assert stats['hit_rate'] == '50.00%'