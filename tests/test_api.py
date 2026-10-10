"""
Integration tests for FastAPI REST API endpoints.
Tests /health, /query, /metrics, /cache/stats, and input validation.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app, service


@pytest.fixture(autouse=True)
def reset_service():
    """Reset cache and metrics before each test."""
    service.cache.invalidate()
    service.cost_tracker.reset()


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "total_cached_entries" in data


def test_query_flow_miss_then_hit(client):
    query = "What is the capital of France?"

    # First request -> Cache Miss
    resp1 = client.post("/query", json={"query": query})
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["cache_hit"] is False
    assert data1["selected_route"] in ["small", "large"]
    assert len(data1["answer"]) > 0

    # Second request with semantic paraphrase -> Cache Hit
    paraphrase = "Can you tell me the capital city of France?"
    resp2 = client.post("/query", json={"query": paraphrase})
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["cache_hit"] is True
    assert data2["selected_route"] == "cache"
    assert data2["estimated_cost_usd"] == 0.0
    assert data2["similarity_score"] >= 0.80


def test_query_validation(client):
    # Empty query should return 422
    resp_empty = client.post("/query", json={"query": "   "})
    assert resp_empty.status_code == 422

    # Invalid force_route should return 400
    resp_invalid_route = client.post(
        "/query", json={"query": "Hello", "force_route": "super_mega_model"}
    )
    assert resp_invalid_route.status_code == 400


def test_cache_stats_and_delete(client):
    # Store a query
    client.post("/query", json={"query": "What is Python?"})

    stats_resp = client.get("/cache/stats")
    assert stats_resp.status_code == 200
    assert stats_resp.json()["total_cached_entries"] >= 1

    # Delete cache
    del_resp = client.delete("/cache")
    assert del_resp.status_code == 200

    # Verify emptied
    stats_after = client.get("/cache/stats")
    assert stats_after.json()["total_cached_entries"] == 0


def test_metrics_endpoint(client):
    client.post("/query", json={"query": "What is 10 * 10?"})
    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    data = metrics_resp.json()
    assert data["total_requests"] >= 1
    assert "cache_hit_rate" in data
    assert "mean_latency_ms" in data
