"""
Streamlit Monitoring Dashboard for SmartCache ML.
Displays real-time metrics, cache statistics, policy comparison,
and provides an interactive query test interface.
"""

import time
import pandas as pd
import numpy as np
import streamlit as st
from smartcache.config import settings
from smartcache.service import SmartCacheService

# Page configuration
st.set_page_config(
    page_title="SmartCache ML Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize service in session state
if "service" not in st.session_state:
    st.session_state.service = SmartCacheService()
    # Pre-populate with a few initial demo queries if cache is empty
    if len(st.session_state.service.cache._ids) == 0:
        st.session_state.service.process_query("What is the capital of France?")
        st.session_state.service.process_query("Can you tell me France's capital city?")
        st.session_state.service.process_query("Explain QuickSort algorithm step by step.")

service: SmartCacheService = st.session_state.service

# ---------------------------------------------------------------------------
# Sidebar Controls & System Status
# ---------------------------------------------------------------------------
st.sidebar.title("⚡ SmartCache ML")
st.sidebar.caption("Semantic Caching & Cost-Aware LLM Routing")

st.sidebar.subheader("System Configuration")
st.sidebar.write(f"**Provider:** `{settings.llm_provider}`")
st.sidebar.write(f"**Embedding Model:** `{settings.embedding_model_name.split('/')[-1]}`")
st.sidebar.write(f"**Similarity Threshold:** `{service.cache.similarity_threshold}`")
st.sidebar.write(f"**Max Cache Size:** `{service.cache.max_cache_size}`")

st.sidebar.markdown("---")
st.sidebar.subheader("Cache Management")
if st.sidebar.button("🧹 Clear / Invalidate Cache", use_container_width=True):
    service.cache.invalidate()
    service.cost_tracker.reset()
    st.sidebar.success("Cache and metrics reset!")
    st.rerun()

# ---------------------------------------------------------------------------
# Main Header & High-Level KPIs
# ---------------------------------------------------------------------------
st.title("Semantic Caching & LLM Routing Analytics")
st.markdown(
    "Live observability into language model inference savings, latency distributions, and query complexity routing."
)

summary = service.cost_tracker.get_summary()
total_req = summary["total_requests"]
hit_rate = summary["cache_hit_rate"] * 100.0
total_cost = summary["total_cost_usd"]
mean_lat = summary["mean_latency_ms"]
p95_lat = summary["p95_latency_ms"]

# Baseline comparison estimate (Hypothetical No-Cache Large Model cost: ~$0.0035/req)
hypothetical_baseline_cost = total_req * 0.0035
savings_pct = (
    ((hypothetical_baseline_cost - total_cost) / hypothetical_baseline_cost) * 100.0
    if hypothetical_baseline_cost > 0
    else 0.0
)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Requests", f"{total_req}")
col2.metric("Cache Hit Rate", f"{hit_rate:.1f}%", delta=f"{summary['cache_hits']} hits")
col3.metric("Est. Total Cost", f"${total_cost:.5f}", delta=f"-{savings_pct:.1f}% saved", delta_color="inverse")
col4.metric("Mean Latency", f"{mean_lat:.1f} ms")
col5.metric("P95 Tail Latency", f"{p95_lat:.1f} ms")

st.markdown("---")

# ---------------------------------------------------------------------------
# Interactive Query Playground
# ---------------------------------------------------------------------------
st.subheader("🧪 Live Query Playground")
with st.form("query_form", clear_on_submit=False):
    user_query = st.text_input(
        "Enter a prompt or question:",
        value="What is the capital of France?",
        placeholder="Type a query to observe caching and routing..."
    )
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        namespace_input = st.text_input("Namespace (isolation context):", value="default")
    with col_opt2:
        bypass = st.checkbox("Bypass Cache (Force inference)", value=False)
    with col_opt3:
        cacheable_flag = st.checkbox("Cacheable Response", value=True)

    submitted = st.form_submit_button("Submit Query", use_container_width=True)

if submitted and user_query.strip():
    result = service.process_query(
        query=user_query.strip(),
        namespace=namespace_input.strip() or "default",
        bypass_cache=bypass,
        is_cacheable=cacheable_flag
    )

    if result.cache_hit:
        st.success(
            f"⚡ **Cache Hit!** Similarity score: **{result.similarity_score:.4f}** | "
            f"Latency: **{result.latency_ms:.1f} ms** | Cost: **$0.000000**"
        )
    else:
        st.info(
            f"🚀 **Cache Miss.** Routed to: **{result.selected_route.upper()}** model | "
            f"Latency: **{result.latency_ms:.1f} ms** | Est. Cost: **${result.estimated_cost_usd:.6f}**"
        )

    st.write(f"**Reasoning:** {result.routing_reason}")
    st.text_area("Answer:", value=result.answer, height=90, disabled=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Analytics Charts & Distributions
# ---------------------------------------------------------------------------
st.subheader("📊 Traffic & Cost Telemetry")
chart_col1, chart_col2 = st.columns(2)

records = service.cost_tracker.records

if records:
    df_records = pd.DataFrame([
        {
            "Request ID": r.request_id,
            "Query": r.query_text,
            "Cache Hit": "Hit" if r.cache_hit else "Miss",
            "Route": r.selected_route,
            "Cost ($)": r.estimated_cost_usd,
            "Latency (ms)": r.latency_seconds * 1000.0,
            "Embedding (ms)": r.embedding_latency * 1000.0,
            "Search (ms)": r.cache_lookup_latency * 1000.0,
            "Generation (ms)": r.generation_latency * 1000.0,
        }
        for r in records
    ])

    with chart_col1:
        st.write("**Route Distribution**")
        route_counts = df_records["Route"].value_counts()
        st.bar_chart(route_counts)

    with chart_col2:
        st.write("**Latency vs Cost by Request**")
        st.scatter_chart(df_records, x="Latency (ms)", y="Cost ($)", color="Route")

    st.markdown("---")
    st.subheader("📜 Recent Request Audit Log")

    # Filter controls
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        route_filter = st.multiselect("Filter by Route:", options=df_records["Route"].unique().tolist(), default=df_records["Route"].unique().tolist())
    with f_col2:
        hit_filter = st.multiselect("Filter by Cache Status:", options=["Hit", "Miss"], default=["Hit", "Miss"])

    filtered_df = df_records[
        (df_records["Route"].isin(route_filter)) &
        (df_records["Cache Hit"].isin(hit_filter))
    ]

    st.dataframe(filtered_df, use_container_width=True)
else:
    st.info("No queries processed yet. Submit queries above to visualize telemetry.")

st.markdown("---")
st.caption(
    "SmartCache ML Portfolio Project | Educational Demonstration | Zero Paid APIs Required"
)
