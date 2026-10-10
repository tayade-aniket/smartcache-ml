"""
Unit tests for CostTracker and metrics calculations.
"""

import pytest
from smartcache.cost_tracker import CostTracker, RequestRecord
from smartcache.metrics import calculate_token_overlap_f1, compare_policies


def test_cost_calculation():
    # Configure custom pricing
    tracker = CostTracker(
        price_in_small=0.0001,
        price_out_small=0.0005,
        price_in_large=0.001,
        price_out_large=0.005,
    )

    # 1. Cache hit -> 0.0 USD
    cost_cached = tracker.calculate_cost(
        route="cache", input_tokens=100, output_tokens=100, is_cache_hit=True
    )
    assert cost_cached == 0.0

    # 2. Small model: 1,000 in, 1,000 out -> 0.0001 + 0.0005 = 0.0006
    cost_small = tracker.calculate_cost(
        route="small", input_tokens=1000, output_tokens=1000, is_cache_hit=False
    )
    assert pytest.approx(cost_small, 1e-6) == 0.0006

    # 3. Large model: 1,000 in, 1,000 out -> 0.001 + 0.005 = 0.006
    cost_large = tracker.calculate_cost(
        route="large", input_tokens=1000, output_tokens=1000, is_cache_hit=False
    )
    assert pytest.approx(cost_large, 1e-6) == 0.006


def test_tracker_summary_metrics():
    tracker = CostTracker()
    tracker.reset()

    # Record 1 hit and 1 miss
    r1 = RequestRecord(
        request_id="r1",
        query_text="What is Paris?",
        cache_hit=True,
        similarity_score=0.92,
        selected_route="cache",
        is_mock=True,
        input_tokens=0,
        output_tokens=0,
        latency_seconds=0.005,
        estimated_cost_usd=0.0,
    )
    r2 = RequestRecord(
        request_id="r2",
        query_text="Explain quantum mechanics.",
        cache_hit=False,
        similarity_score=None,
        selected_route="large",
        is_mock=True,
        input_tokens=500,
        output_tokens=500,
        latency_seconds=0.050,
        estimated_cost_usd=0.00375,
    )

    tracker.record_request(r1)
    tracker.record_request(r2)

    summary = tracker.get_summary()
    assert summary["total_requests"] == 2
    assert summary["cache_hits"] == 1
    assert summary["cache_misses"] == 1
    assert summary["cache_hit_rate"] == 0.5
    assert summary["total_cost_usd"] == 0.00375
    assert summary["mean_latency_ms"] == pytest.approx(27.5, 0.1)


def test_quality_and_policy_comparison():
    # Exact token overlap
    f1_perfect = calculate_token_overlap_f1("The capital of France is Paris.", "The capital of France is Paris.")
    assert f1_perfect == 1.0

    # Partial overlap
    f1_partial = calculate_token_overlap_f1("Paris is France's capital city.", "The capital of France is Paris.")
    assert 0.0 < f1_partial < 1.0

    # Disjoint tokens
    f1_zero = calculate_token_overlap_f1("Apple orange banana", "Quantum computing superposition")
    assert f1_zero == 0.0

    # Policy comparison
    baseline = {"total_cost_usd": 10.0, "mean_latency_ms": 200.0, "p95_latency_ms": 400.0}
    optimized = {"total_cost_usd": 3.0, "mean_latency_ms": 80.0, "p95_latency_ms": 150.0, "cache_hit_rate": 0.65}

    cmp = compare_policies(baseline, optimized)
    assert cmp["cost_savings_pct"] == 70.0
    assert cmp["latency_reduction_pct"] == 60.0
    assert cmp["p95_latency_reduction_pct"] == 62.5
