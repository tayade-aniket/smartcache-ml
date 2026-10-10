"""
Cost Tracker Module for SmartCache ML.
Tracks per-request inference metrics, token counts, latency, and estimated monetary cost.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
from smartcache.config import settings


@dataclass
class RequestRecord:
    request_id: str
    query_text: str
    cache_hit: bool
    similarity_score: Optional[float]
    selected_route: str  # 'cache', 'small', 'large'
    is_mock: bool
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    embedding_latency: float = 0.0
    cache_lookup_latency: float = 0.0
    generation_latency: float = 0.0
    estimated_cost_usd: float = 0.0
    reference_answer: Optional[str] = None
    generated_answer: str = ""
    quality_score: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CostTracker:
    """
    Accumulates request records and computes cost and latency metrics.
    """

    def __init__(
        self,
        price_in_small: float = settings.price_per_1k_input_tokens_small,
        price_out_small: float = settings.price_per_1k_output_tokens_small,
        price_in_large: float = settings.price_per_1k_input_tokens_large,
        price_out_large: float = settings.price_per_1k_output_tokens_large,
    ):
        import threading
        self.price_in_small = price_in_small
        self.price_out_small = price_out_small
        self.price_in_large = price_in_large
        self.price_out_large = price_out_large
        self.records: List[RequestRecord] = []
        self._lock = threading.Lock()

    def calculate_cost(self, route: str, input_tokens: int, output_tokens: int, is_cache_hit: bool) -> float:
        """
        Calculate estimated cloud API inference cost in USD.
        Cached queries incur 0.0 LLM inference cost.
        """
        if is_cache_hit:
            return 0.0

        if route == "small":
            return (input_tokens / 1000.0) * self.price_in_small + (output_tokens / 1000.0) * self.price_out_small
        elif route == "large":
            return (input_tokens / 1000.0) * self.price_in_large + (output_tokens / 1000.0) * self.price_out_large
        return 0.0

    def record_request(self, record: RequestRecord) -> RequestRecord:
        """Calculate cost if not already populated and store the record."""
        if record.estimated_cost_usd == 0.0 and not record.cache_hit:
            record.estimated_cost_usd = self.calculate_cost(
                route=record.selected_route,
                input_tokens=record.input_tokens,
                output_tokens=record.output_tokens,
                is_cache_hit=record.cache_hit,
            )
        with self._lock:
            self.records.append(record)
        return record

    def reset(self):
        """Clear recorded requests."""
        with self._lock:
            self.records.clear()

    def get_summary(self) -> Dict[str, Any]:
        """Compute aggregate summary metrics over all recorded requests."""
        with self._lock:
            if not self.records:
                return {
                    "total_requests": 0,
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "cache_hit_rate": 0.0,
                    "total_cost_usd": 0.0,
                    "mean_cost_per_request": 0.0,
                    "mean_latency_ms": 0.0,
                    "p50_latency_ms": 0.0,
                    "p95_latency_ms": 0.0,
                    "route_distribution": {},
                    "error_count": 0,
                }

            total = len(self.records)
            hits = sum(1 for r in self.records if r.cache_hit)
            misses = total - hits
            costs = [r.estimated_cost_usd for r in self.records]
            latencies_ms = [r.latency_seconds * 1000.0 for r in self.records]
            errors = sum(1 for r in self.records if r.error is not None)

            routes: Dict[str, int] = {}
            for r in self.records:
                routes[r.selected_route] = routes.get(r.selected_route, 0) + 1

        return {
            "total_requests": total,
            "cache_hits": hits,
            "cache_misses": misses,
            "cache_hit_rate": hits / total if total > 0 else 0.0,
            "total_cost_usd": float(np.sum(costs)),
            "mean_cost_per_request": float(np.mean(costs)),
            "mean_latency_ms": float(np.mean(latencies_ms)),
            "p50_latency_ms": float(np.percentile(latencies_ms, 50)),
            "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
            "route_distribution": routes,
            "error_count": errors,
        }
