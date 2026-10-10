"""
SmartCache Orchestration Service.
Coordinates semantic caching, query complexity routing, LLM generation, and cost tracking.
"""

import time
import uuid
from dataclasses import dataclass
from typing import Optional, Dict, Any
from smartcache.config import settings
from smartcache.cache import SemanticCache, CacheLookupResult
from smartcache.router import ClassifierRouter, RoutingDecision
from smartcache.llm_provider import BaseLLMProvider, get_llm_provider
from smartcache.cost_tracker import CostTracker, RequestRecord
from smartcache.embeddings import encode_query


@dataclass
class ServiceResult:
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


class SmartCacheService:
    """
    Main entry point service orchestrating caching, routing, inference, and metric logging.
    """

    def __init__(
        self,
        cache: Optional[SemanticCache] = None,
        router: Optional[ClassifierRouter] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        cost_tracker: Optional[CostTracker] = None,
    ):
        self.cache = cache or SemanticCache()
        self.router = router or ClassifierRouter()
        self.provider = llm_provider or get_llm_provider(settings.llm_provider)
        self.cost_tracker = cost_tracker or CostTracker()

    def check_cache_only(self, query: str, namespace: str = "default") -> CacheLookupResult:
        """Perform a lookup in the cache without triggering generation or fallback."""
        return self.cache.lookup(query, namespace=namespace)

    def process_query(
        self,
        query: str,
        namespace: str = "default",
        bypass_cache: bool = False,
        force_route: Optional[str] = None,
        is_cacheable: bool = True,
    ) -> ServiceResult:
        req_id = f"req_{uuid.uuid4().hex[:8]}"
        overall_start = time.perf_counter()

        embedding_latency_ms = 0.0
        cache_lookup_latency_ms = 0.0
        generation_latency_ms = 0.0
        similarity_score = None
        cache_hit = False
        answer = ""
        route_decision = None
        selected_route = "cache"
        routing_reason = "Answer retrieved from semantic cache"
        is_mock = False
        input_tokens = 0
        output_tokens = 0

        # Step 1: Cache Lookup (if enabled for this query)
        if not bypass_cache and is_cacheable:
            lookup_res: CacheLookupResult = self.cache.lookup(query, namespace=namespace)
            embedding_latency_ms = lookup_res.embedding_latency * 1000.0
            cache_lookup_latency_ms = lookup_res.search_latency * 1000.0
            similarity_score = lookup_res.similarity_score

            if lookup_res.hit:
                cache_hit = True
                answer = lookup_res.response
                selected_route = "cache"
                routing_reason = (
                    f"Cache Hit! Matched '{lookup_res.matched_query}' "
                    f"with similarity score {lookup_res.similarity_score:.4f}"
                )
                total_latency = time.perf_counter() - overall_start

                # Log cached request in cost tracker
                record = RequestRecord(
                    request_id=req_id,
                    query_text=query,
                    cache_hit=True,
                    similarity_score=similarity_score,
                    selected_route="cache",
                    is_mock=False,
                    input_tokens=0,
                    output_tokens=0,
                    latency_seconds=total_latency,
                    embedding_latency=lookup_res.embedding_latency,
                    cache_lookup_latency=lookup_res.search_latency,
                    generation_latency=0.0,
                    estimated_cost_usd=0.0,
                    generated_answer=answer,
                )
                self.cost_tracker.record_request(record)

                return ServiceResult(
                    request_id=req_id,
                    query=query,
                    answer=answer,
                    cache_hit=True,
                    similarity_score=similarity_score,
                    selected_route=selected_route,
                    routing_reason=routing_reason,
                    latency_ms=round(total_latency * 1000.0, 2),
                    estimated_cost_usd=0.0,
                    is_mock=False,
                    embedding_latency_ms=round(embedding_latency_ms, 2),
                    cache_lookup_latency_ms=round(cache_lookup_latency_ms, 2),
                    generation_latency_ms=0.0,
                    metadata={"matched_query": lookup_res.matched_query, "cache_id": lookup_res.cache_id}
                )

        # Step 2: Cache Miss -> Routing Decision
        if force_route in ["small", "large"]:
            selected_route = force_route
            routing_reason = f"Forced route override: {force_route}"
        else:
            route_decision: RoutingDecision = self.router.route(query)
            selected_route = route_decision.selected_route
            routing_reason = route_decision.reason

        # Step 3: LLM Inference Execution
        t_gen_start = time.perf_counter()
        llm_resp = self.provider.generate(query, model_type=selected_route)
        generation_latency_ms = (time.perf_counter() - t_gen_start) * 1000.0

        answer = llm_resp.text
        is_mock = llm_resp.is_mock
        input_tokens = llm_resp.input_tokens
        output_tokens = llm_resp.output_tokens

        # Step 4: Populate Cache for future hits (if cacheable)
        if is_cacheable:
            query_vec = encode_query(query, normalize=True)
            self.cache.store(query, answer, embedding=query_vec, namespace=namespace, is_cacheable=True)

        total_latency = time.perf_counter() - overall_start

        # Step 5: Record in Cost Tracker
        cost_usd = self.cost_tracker.calculate_cost(
            route=selected_route,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            is_cache_hit=False
        )

        record = RequestRecord(
            request_id=req_id,
            query_text=query,
            cache_hit=False,
            similarity_score=similarity_score,
            selected_route=selected_route,
            is_mock=is_mock,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=total_latency,
            embedding_latency=embedding_latency_ms / 1000.0,
            cache_lookup_latency=cache_lookup_latency_ms / 1000.0,
            generation_latency=generation_latency_ms / 1000.0,
            estimated_cost_usd=cost_usd,
            generated_answer=answer,
        )
        self.cost_tracker.record_request(record)

        return ServiceResult(
            request_id=req_id,
            query=query,
            answer=answer,
            cache_hit=False,
            similarity_score=similarity_score,
            selected_route=selected_route,
            routing_reason=routing_reason,
            latency_ms=round(total_latency * 1000.0, 2),
            estimated_cost_usd=round(cost_usd, 6),
            is_mock=is_mock,
            embedding_latency_ms=round(embedding_latency_ms, 2),
            cache_lookup_latency_ms=round(cache_lookup_latency_ms, 2),
            generation_latency_ms=round(generation_latency_ms, 2),
            metadata={"input_tokens": input_tokens, "output_tokens": output_tokens, "model_name": llm_resp.model_name}
        )
