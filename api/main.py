"""
FastAPI REST API for SmartCache ML.
Provides endpoints for querying, cache inspection, invalidation, and metrics.
"""

from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field
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
    clean_query = req.query.strip()
    if not clean_query:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query string cannot be empty or whitespace only."
        )

    if req.force_route and req.force_route not in ["small", "large"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="force_route must be either 'small' or 'large'."
        )

    try:
        result = service.process_query(
            query=clean_query,
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


@app.get("/metrics")
def get_metrics():
    """Retrieve cumulative request, cost, latency, and hit rate metrics."""
    return service.cost_tracker.get_summary()


@app.get("/cache/stats")
def get_cache_stats():
    """Retrieve current cache statistics."""
    return service.cache.get_stats()


@app.delete("/cache")
def clear_cache(namespace: Optional[str] = Query(None, description="Optional namespace to invalidate")):
    """Invalidate cache entries."""
    service.cache.invalidate(namespace=namespace)
    return {
        "status": "success",
        "message": f"Cache invalidated for namespace: {namespace or 'ALL'}"
    }
