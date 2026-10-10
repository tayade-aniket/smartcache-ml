"""
Integration tests for FastAPI REST API endpoints.
Tests /health, /query, /cache/lookup, /cache/entries, /cache/stats, /metrics, and validation.
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

    # Invalid force_route should fail schema validation with 422
    resp_invalid_route = client.post(
        "/query", json={"query": "Hello", "force_route": "super_mega_model"}
    )
    assert resp_invalid_route.status_code == 422


def test_cache_lookup_endpoint(client):
    query = "What is the capital of France?"

    # Lookup before query is stored -> Cache Miss
    resp_before = client.post("/cache/lookup", json={"query": query})
    assert resp_before.status_code == 200
    assert resp_before.json()["cache_hit"] is False

    # Store query via /query
    client.post("/query", json={"query": query})

    # Lookup with paraphrase -> Cache Hit
    resp_after = client.post(
        "/cache/lookup", json={"query": "Can you tell me France's capital city?"}
    )
    assert resp_after.status_code == 200
    assert resp_after.json()["cache_hit"] is True
    assert resp_after.json()["similarity_score"] >= 0.80
    assert "Paris" in resp_after.json()["cached_response"]


def test_cache_entries_and_delete(client):
    # Store a query
    client.post("/query", json={"query": "What is Python?"})

    # Inspect entries
    entries_resp = client.get("/cache/entries?limit=10")
    assert entries_resp.status_code == 200
    entries = entries_resp.json()
    assert len(entries) >= 1
    assert "query_text" in entries[0]
    assert "embedding" not in entries[0]  # Verify binary blob is excluded

    # Delete cache
    del_resp = client.delete("/cache")
    assert del_resp.status_code == 200

    # Verify emptied
    stats_after = client.get("/cache/stats")
    assert stats_after.json()["total_cached_entries"] == 0


def test_metrics_and_reset_endpoints(client):
    client.post("/query", json={"query": "What is 10 * 10?"})

    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    data = metrics_resp.json()
    assert data["total_requests"] >= 1

    # Reset metrics
    reset_resp = client.post("/metrics/reset")
    assert reset_resp.status_code == 200

    # Verify reset
    metrics_after = client.get("/metrics")
    assert metrics_after.json()["total_requests"] == 0
