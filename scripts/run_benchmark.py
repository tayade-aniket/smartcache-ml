"""
Benchmark Runner Script for SmartCache ML.
Evaluates multi-policy performance (cost, latency, cache hit rate, quality score)
on standardized test workloads and exports reports.
"""

import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from smartcache.config import settings
from smartcache.cache import SemanticCache, ExactMatchCache
from smartcache.router import ClassifierRouter
from smartcache.llm_provider import MockProvider
from smartcache.cost_tracker import CostTracker, RequestRecord
from smartcache.metrics import calculate_token_overlap_f1, compare_policies


def run_benchmark():
    test_path = settings.data_processed_path / "test_queries.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing {test_path}. Run generate_sample_data.py first.")

    test_df = pd.read_csv(test_path)
    # Build realistic streaming workload: test queries followed by paraphrases
    workload = pd.concat([test_df, test_df[test_df["query_role"] == "paraphrase"]]).reset_index(drop=True)
    print(f"Loaded benchmark workload of {len(workload)} requests.")

    policies = [
        {"name": "1. Baseline (Always Large, No Cache)", "sem": False, "exact": False, "route": "always_large"},
        {"name": "2. Exact Cache + Large Model", "sem": False, "exact": True, "route": "always_large"},
        {"name": "3. Semantic Cache + Large Model", "sem": True, "exact": False, "route": "always_large"},
        {"name": "4. Always Small Model (No Cache)", "sem": False, "exact": False, "route": "always_small"},
        {"name": "5. SmartCache ML (Semantic + Router)", "sem": True, "exact": False, "route": "router"},
    ]

    all_results = []
    baseline_summary = None

    for p in policies:
        print(f"Evaluating {p['name']}...")
        tracker = CostTracker()
        provider = MockProvider()
        
        # Fresh isolated DB for test run
        temp_db = settings.project_root / "data" / f"bench_{abs(hash(p['name'])) % 10000}.db"
        try:
            if temp_db.exists():
                temp_db.unlink()
        except Exception:
            pass
            
        sem_cache = SemanticCache(db_path=temp_db, similarity_threshold=0.85)
        exact_cache = ExactMatchCache()
        router = ClassifierRouter()

        for _, row in workload.iterrows():
            query = row["query_text"]
            ref_ans = row["expected_answer"]
            cacheable = row["is_cacheable"]

            hit = False
            ans = ""
            sim_score = None
            lat = 0.0
            route_selected = "large"

            # Cache lookup
            if p["exact"] and cacheable:
                look = exact_cache.lookup(query)
                if look.hit:
                    hit = True
                    ans = look.response
                    lat = look.total_lookup_latency
                    route_selected = "cache"
            elif p["sem"] and cacheable:
                look = sem_cache.lookup(query)
                sim_score = look.similarity_score
                if look.hit:
                    hit = True
                    ans = look.response
                    lat = look.total_lookup_latency
                    route_selected = "cache"

            # Cache miss -> Inference
            if not hit:
                if p["route"] == "always_large":
                    route_selected = "large"
                elif p["route"] == "always_small":
                    route_selected = "small"
                elif p["route"] == "router":
                    dec = router.route(query)
                    route_selected = dec.selected_route

                resp = provider.generate(query, model_type=route_selected)
                ans = resp.text
                lat = resp.latency_seconds

                if cacheable:
                    if p["exact"]:
                        exact_cache.store(query, ans)
                    if p["sem"]:
                        sem_cache.store(query, ans)

            cost = tracker.calculate_cost(route_selected, len(query) // 4, len(ans) // 4, is_cache_hit=hit)
            quality = calculate_token_overlap_f1(ans, ref_ans)

            rec = RequestRecord(
                request_id=f"bench_{row['query_id']}",
                query_text=query,
                cache_hit=hit,
                similarity_score=sim_score,
                selected_route=route_selected,
                is_mock=True,
                input_tokens=len(query) // 4 if not hit else 0,
                output_tokens=len(ans) // 4 if not hit else 0,
                latency_seconds=lat,
                estimated_cost_usd=cost,
                reference_answer=ref_ans,
                generated_answer=ans,
                quality_score=quality,
            )
            tracker.record_request(rec)

        summary = tracker.get_summary()
        mean_quality = float(np.mean([r.quality_score for r in tracker.records]))
        summary["mean_quality_f1"] = round(mean_quality, 4)

        if p["route"] == "always_large" and not p["sem"] and not p["exact"]:
            baseline_summary = summary

        comparison = compare_policies(baseline_summary, summary) if baseline_summary else {}

        all_results.append({
            "Policy": p["name"],
            "Requests": summary["total_requests"],
            "Cache Hit Rate (%)": round(summary["cache_hit_rate"] * 100.0, 1),
            "Total Cost ($)": round(summary["total_cost_usd"], 5),
            "Cost Savings (%)": comparison.get("cost_savings_pct", 0.0),
            "Mean Latency (ms)": round(summary["mean_latency_ms"], 1),
            "P50 Latency (ms)": round(summary["p50_latency_ms"], 1),
            "P95 Latency (ms)": round(summary["p95_latency_ms"], 1),
            "Latency Reduction (%)": comparison.get("latency_reduction_pct", 0.0),
            "Quality Score (F1)": summary["mean_quality_f1"],
        })

        # Cleanup temp db
        sem_cache.invalidate()
        try:
            if temp_db.exists():
                temp_db.unlink()
        except Exception:
            pass

    df_out = pd.DataFrame(all_results)
    print("\n" + "=" * 90)
    print("SMARTCACHE ML BENCHMARK EVALUATION RESULTS")
    print("=" * 90)
    print(df_out.to_string(index=False))
    print("=" * 90)

    # Save reports
    report_csv = settings.reports_path / "benchmark_report.csv"
    report_json = settings.reports_path / "benchmark_report.json"
    df_out.to_csv(report_csv, index=False)
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved benchmark CSV to: {report_csv}")
    print(f"Saved benchmark JSON to: {report_json}")


if __name__ == "__main__":
    run_benchmark()
