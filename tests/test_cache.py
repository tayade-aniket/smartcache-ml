"""
Unit tests for ExactMatchCache and SemanticCache.
Tests exact hits, semantic similarity hits, threshold rejection,
namespace isolation, expiration TTL, and non-cacheable entries.
"""

import time
import pytest
import numpy as np
from pathlib import Path
from smartcache.cache import ExactMatchCache, SemanticCache


def test_exact_match_cache():
    cache = ExactMatchCache(ttl_seconds=10)
    cache.store("What is the capital of France?", "Paris", is_cacheable=True)

    # Exact match hit (case-insensitive & stripped)
    res = cache.lookup("what is the capital of france?  ")
    assert res.hit is True
    assert res.response == "Paris"
    assert res.similarity_score == 1.0

    # Miss for paraphrase in exact cache
    res_miss = cache.lookup("Can you tell me the capital city of France?")
    assert res_miss.hit is False

    # Non-cacheable query
    stored = cache.store("What time is it?", "12:00 PM", is_cacheable=False)
    assert stored is False
    assert cache.lookup("What time is it?").hit is False


def test_semantic_cache_creation_and_hit(tmp_path):
    db_file = tmp_path / "test_cache.db"
    cache = SemanticCache(db_path=db_file, similarity_threshold=0.80)

    query1 = "What is the capital of France?"
    ans1 = "The capital of France is Paris."
    cache.store(query1, ans1, is_cacheable=True)

    # Identical query
    res1 = cache.lookup(query1)
    assert res1.hit is True
    assert res1.similarity_score > 0.99
    assert res1.response == ans1

    # Semantic paraphrase hit
    paraphrase = "Can you tell me France's capital city?"
    res2 = cache.lookup(paraphrase)
    assert res2.hit is True
    assert res2.similarity_score >= 0.80
    assert res2.response == ans1

    # Unrelated query rejection
    unrelated = "How do you calculate compound interest on an investment?"
    res3 = cache.lookup(unrelated)
    assert res3.hit is False
    assert res3.similarity_score < 0.60


def test_semantic_cache_threshold_behavior(tmp_path):
    db_file = tmp_path / "threshold_test.db"
    # Strict threshold 0.95
    cache = SemanticCache(db_path=db_file, similarity_threshold=0.95)

    cache.store("What is the speed of light in vacuum?", "299,792,458 m/s")

    # Loose paraphrase might be ~0.88, should miss with threshold 0.95
    loose_query = "Tell me how quickly photons travel across empty space"
    res = cache.lookup(loose_query)
    # Check that threshold strictly blocks it
    if res.similarity_score < 0.95:
        assert res.hit is False

    # But with threshold override of 0.70 it should hit
    res_relaxed = cache.lookup(loose_query, threshold_override=0.70)
    if res.similarity_score >= 0.70:
        assert res_relaxed.hit is True


def test_namespace_isolation(tmp_path):
    db_file = tmp_path / "ns_test.db"
    cache = SemanticCache(db_path=db_file, similarity_threshold=0.80)

    # Store entry in user_alice namespace
    cache.store(
        query="What is my confidential account balance?",
        response="Your balance is $5,000",
        namespace="user_alice"
    )

    # Lookup from user_bob namespace MUST NOT return Alice's confidential data
    res_bob = cache.lookup("What is my confidential account balance?", namespace="user_bob")
    assert res_bob.hit is False

    # Alice lookup succeeds
    res_alice = cache.lookup("What is my confidential account balance?", namespace="user_alice")
    assert res_alice.hit is True
    assert res_alice.response == "Your balance is $5,000"


def test_cache_expiration_and_invalidation(tmp_path):
    db_file = tmp_path / "exp_test.db"
    # Short TTL of 1 second
    cache = SemanticCache(db_path=db_file, ttl_seconds=1)

    cache.store("What is the freezing point of water?", "0 degrees Celsius")
    assert cache.lookup("What is the freezing point of water?").hit is True

    # Sleep 1.1s for expiration
    time.sleep(1.1)

    # Re-sync / lookup should detect expiration
    cache._sync_from_db()
    res_expired = cache.lookup("What is the freezing point of water?")
    assert res_expired.hit is False

    # Test explicit invalidation
    cache.store("Another question", "Another answer")
    assert cache.lookup("Another question").hit is True
    cache.invalidate()
    assert cache.lookup("Another question").hit is False
