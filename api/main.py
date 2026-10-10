"""
FastAPI REST API for SmartCache ML.
Provides endpoints for querying, cache inspection, invalidation, and metrics.
"""

from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator
from smartcache.config import settings
from smartcache.service import SmartCacheService

app = FastAPI(
    title="SmartCache ML API",
    description="Semantic Caching and Cost-Aware LLM Routing REST Service",
    version="0.1.0",
)

# Global service instance
service = SmartCacheService()


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="The user query or prompt")
    namespace: str = Field(default="default", max_length=100, description="Cache isolation namespace")
    bypass_cache: bool = Field(default=False, description="Whether to bypass cache lookup")
    force_route: Optional[str] = Field(default=None, description="Force 'small' or 'large' model route")
    is_cacheable: bool = Field(default=True, description="Whether the response can be stored in cache")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Query string cannot be empty or whitespace only.")
        return clean

    @field_validator("force_route")
    @classmethod
    def validate_force_route(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ["small", "large"]:
            raise ValueError("force_route must be either 'small' or 'large'.")
        return v


class QueryResponse(BaseModel):
    request_id: str
    query: str
    answer: str
    cache_hit: bool
    similarity_score: Optional[float]
    selected_route: str
    routing_reason: str
    latency_ms: float
    estimated_cost_usd: float
    is_mock: bool
    embedding_latency_ms: float
    cache_lookup_latency_ms: float
    generation_latency_ms: float
    metadata: Dict[str, Any]


class CacheLookupRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Query to look up in cache")
    namespace: str = Field(default="default", max_length=100, description="Cache isolation namespace")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Query string cannot be empty or whitespace only.")
        return clean


class CacheLookupResponse(BaseModel):
    query: str
    namespace: str
    cache_hit: bool
    matched_query: Optional[str] = None
    cached_response: Optional[str] = None
    similarity_score: Optional[float] = None
    cache_id: Optional[int] = None
    lookup_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    provider: str
    total_cached_entries: int
    similarity_threshold: float


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Service health and current cache size."""
    stats = service.cache.get_stats()
    return HealthResponse(
        status="healthy",
        provider=settings.llm_provider,
        total_cached_entries=stats["total_cached_entries"],
        similarity_threshold=stats["similarity_threshold"]
    )


@app.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
def process_query_endpoint(req: QueryRequest):
    """
    Submit a query to the SmartCache service.
    Performs semantic lookup -> routing -> generation -> cost tracking.
    """
    try:
        result = service.process_query(
            query=req.query,
            namespace=req.namespace,
            bypass_cache=req.bypass_cache,
            force_route=req.force_route,
            is_cacheable=req.is_cacheable
        )
        return QueryResponse(
            request_id=result.request_id,
            query=result.query,
            answer=result.answer,
            cache_hit=result.cache_hit,
            similarity_score=result.similarity_score,
            selected_route=result.selected_route,
            routing_reason=result.routing_reason,
            latency_ms=result.latency_ms,
            estimated_cost_usd=result.estimated_cost_usd,
            is_mock=result.is_mock,
            embedding_latency_ms=result.embedding_latency_ms,
            cache_lookup_latency_ms=result.cache_lookup_latency_ms,
            generation_latency_ms=result.generation_latency_ms,
            metadata=result.metadata
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


@app.post("/cache/lookup", response_model=CacheLookupResponse)
def cache_lookup_only_endpoint(req: CacheLookupRequest):
    """
    Inspect the semantic cache without triggering LLM generation or incurring costs.
    """
    lookup_result = service.check_cache_only(query=req.query, namespace=req.namespace)
    return CacheLookupResponse(
        query=req.query,
        namespace=req.namespace,
        cache_hit=lookup_result.hit,
        matched_query=lookup_result.matched_query,
        cached_response=lookup_result.response,
        similarity_score=lookup_result.similarity_score,
        cache_id=lookup_result.cache_id,
        lookup_latency_ms=round(lookup_result.total_lookup_latency * 1000.0, 2)
    )


@app.get("/cache/entries", response_model=List[Dict[str, Any]])
def list_cache_entries(
    namespace: Optional[str] = Query(None, description="Optional namespace filter"),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of entries to return")
):
    """Retrieve recent cached entries for inspection (excluding raw binary embeddings)."""
    return service.cache.get_entries(namespace=namespace, limit=limit)


@app.get("/cache/stats")
def get_cache_stats():
    """Retrieve current cache statistics (hits, misses, hit rate, capacity)."""
    return service.cache.get_stats()


@app.delete("/cache")
def clear_cache(namespace: Optional[str] = Query(None, description="Optional namespace to invalidate")):
    """Invalidate all cache entries or entries belonging to a specified namespace."""
    service.cache.invalidate(namespace=namespace)
    return {
        "status": "success",
        "message": f"Cache invalidated for namespace: {namespace or 'ALL'}"
    }


@app.get("/metrics")
def get_metrics():
    """Retrieve cumulative request, cost, latency, and hit rate metrics."""
    return service.cost_tracker.get_summary()


@app.post("/metrics/reset")
def reset_metrics():
    """Reset accumulated request metrics and cost counters."""
    service.cost_tracker.reset()
    return {
        "status": "success",
        "message": "Metrics and cost counters have been reset."
    }
