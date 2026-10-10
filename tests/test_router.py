"""
Unit tests for Query Complexity Routing.
Tests RuleBasedRouter, ClassifierRouter, probability thresholding,
and missing model fallback.
"""

import pytest
from smartcache.router import RuleBasedRouter, ClassifierRouter


def test_rule_based_router():
    router = RuleBasedRouter(length_threshold_words=10)

    # Short simple query
    dec_simple = router.route("What is the capital of Canada?")
    assert dec_simple.selected_route == "small"
    assert dec_simple.method == "rule_based"

    # Trigger keyword: 'derive' / 'step by step'
    dec_complex_kw = router.route("Explain step by step how QuickSort partitions elements.")
    assert dec_complex_kw.selected_route == "large"

    # Long query (> 10 words)
    dec_complex_len = router.route(
        "A system architect wants to build a distributed caching mechanism using write-through policies across multiple geographically isolated database replicas."
    )
    assert dec_complex_len.selected_route == "large"


def test_classifier_router_training_and_decision(tmp_path):
    model_file = tmp_path / "temp_router.joblib"
    router = ClassifierRouter(model_path=model_file, probability_threshold=0.50)

    # Train miniature dataset
    queries = [
        "What is the capital of Italy?",
        "What is 2 + 2?",
        "Define list in Python.",
        "Derive backpropagation with chain rule and mathematical proof.",
        "Explain distributed consensus in Raft and Paxos algorithms.",
        "Step by step architectural design for high throughput microservices."
    ]
    labels = ["simple", "simple", "simple", "complex", "complex", "complex"]

    router.train(queries, labels)
    router.save()

    # Verify saved model reloads
    reloaded_router = ClassifierRouter(model_path=model_file)
    assert reloaded_router.pipeline is not None

    # Test route for simple
    dec1 = reloaded_router.route("What is the capital of Spain?")
    assert dec1.selected_route in ["small", "large"]
    assert dec1.method == "classifier"

    # Test route for complex
    dec2 = reloaded_router.route("Explain distributed consensus algorithms step by step.")
    assert dec2.selected_route == "large"
    assert dec2.method == "classifier"


def test_classifier_missing_model_fallback(tmp_path):
    missing_path = tmp_path / "non_existent_model.joblib"
    router = ClassifierRouter(model_path=missing_path)

    # Without a trained model, it falls back gracefully to rule-based routing
    decision = router.route("Simple query")
    assert decision.selected_route in ["small", "large"]
    assert decision.method == "fallback_rule_based"
