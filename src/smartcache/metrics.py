"""
Evaluation Metrics for SmartCache ML.
Calculates cache effectiveness, answer similarity/quality scores, and comparative savings.
"""

from typing import Dict, Any, List, Set
import re
import numpy as np


def tokenize_simple(text: str) -> Set[str]:
    """Basic alphanumeric tokenization for lightweight quality scoring."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    return set(tokens)


def calculate_token_overlap_f1(prediction: str, reference: str) -> float:
    """
    Calculate token-level F1 (unigram overlap) between prediction and reference answer.
    Ranges between 0.0 (no overlap) to 1.0 (exact token match).
    """
    if not prediction or not reference:
        return 0.0

    pred_tokens = tokenize_simple(prediction)
    ref_tokens = tokenize_simple(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = pred_tokens.intersection(ref_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    f1 = 2.0 * (precision * recall) / (precision + recall)
    return float(f1)


def compare_policies(baseline: Dict[str, Any], optimized: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare an optimized policy (e.g. Semantic Cache + Router) against a baseline (e.g. Always Large No Cache).
    
    Returns:
        cost_savings_pct: % reduction in total cost
        latency_reduction_pct: % reduction in mean latency
        p95_latency_reduction_pct: % reduction in p95 latency
        cache_hit_rate: hit rate achieved by optimized policy
    """
    base_cost = baseline.get("total_cost_usd", 0.0)
    opt_cost = optimized.get("total_cost_usd", 0.0)
    cost_savings_pct = (
        ((base_cost - opt_cost) / base_cost) * 100.0 if base_cost > 0 else 0.0
    )

    base_lat = baseline.get("mean_latency_ms", 0.0)
    opt_lat = optimized.get("mean_latency_ms", 0.0)
    latency_reduction_pct = (
        ((base_lat - opt_lat) / base_lat) * 100.0 if base_lat > 0 else 0.0
    )

    base_p95 = baseline.get("p95_latency_ms", 0.0)
    opt_p95 = optimized.get("p95_latency_ms", 0.0)
    p95_latency_reduction_pct = (
        ((base_p95 - opt_p95) / base_p95) * 100.0 if base_p95 > 0 else 0.0
    )

    return {
        "baseline_cost_usd": base_cost,
        "optimized_cost_usd": opt_cost,
        "cost_savings_pct": round(cost_savings_pct, 2),
        "baseline_mean_lat_ms": round(base_lat, 2),
        "optimized_mean_lat_ms": round(opt_lat, 2),
        "latency_reduction_pct": round(latency_reduction_pct, 2),
        "baseline_p95_lat_ms": round(base_p95, 2),
        "optimized_p95_lat_ms": round(opt_p95, 2),
        "p95_latency_reduction_pct": round(p95_latency_reduction_pct, 2),
        "cache_hit_rate": round(optimized.get("cache_hit_rate", 0.0) * 100.0, 2),
    }
