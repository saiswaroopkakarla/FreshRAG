"""
FreshRAG -- Streamlit demo UI.

This is a thin client: it calls the FastAPI backend's /query endpoint
and visualizes the results (answer + per-source score breakdown). All
the actual logic lives in the FastAPI service, so this file stays
small and easy to restyle.

Run with:
    streamlit run streamlit_app.py

Make sure the backend is already running (see README) -- by default
this expects it at http://localhost:8000 (override with BACKEND_URL
in .env).
"""

import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="FreshRAG", page_icon="🕒", layout="wide")

st.title("🕒 FreshRAG")
st.caption(
    "Adaptive Multi-Source Temporal-Aware Hybrid RAG — ranks live web results "
    "by relevance, freshness, authority, and credibility, not just similarity."
)

with st.sidebar:
    st.header("Backend")
    st.write(f"API: `{BACKEND_URL}`")
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if health.ok:
            st.success("Backend is reachable ✅")
            cfg = requests.get(f"{BACKEND_URL}/config", timeout=3).json()
            st.subheader("Active configuration")
            st.json(cfg)
        else:
            st.error("Backend responded with an error.")
    except requests.RequestException:
        st.error("Backend not reachable. Start it with:\n\nuvicorn app.main:app --reload")

    st.divider()
    top_k = st.slider("Number of sources to rank", min_value=3, max_value=20, value=8)

query = st.text_input(
    "Ask a time-sensitive question",
    placeholder="e.g. Why is Apple stock falling today?",
)

col_run, col_analyze = st.columns([1, 1])
run_clicked = col_run.button("Run full pipeline", type="primary", use_container_width=True)
analyze_clicked = col_analyze.button("Just analyze query", use_container_width=True)

if analyze_clicked and query:
    try:
        resp = requests.post(f"{BACKEND_URL}/analyze", json={"query": query}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        method_label = f"{data['method']}" + (f" ({data['provider']})" if data.get("provider") else "")
        st.subheader(f"Query understanding — via {method_label}")
        st.json(
            {
                "domain/topic": data["domain"],
                "time_sensitive": data["time_sensitive"],
                "intent": data["intent"],
                "keywords": data["keywords"],
                "search_query": data["search_query"],
            }
        )
        st.subheader("Weights generated")
        st.json(data["weights"])
    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")

if run_clicked and query:
    with st.spinner("Searching the web, scoring, and ranking sources..."):
        try:
            resp = requests.post(
                f"{BACKEND_URL}/query", json={"query": query, "top_k": top_k}, timeout=90
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")
            data = None

    if data:
        st.subheader("Answer")
        st.info(data["answer"])
        st.caption(
            f"Generated via: {data['generation_provider']}  |  "
            f"Understood via: {data['understanding_method']}"
            + (f" ({data['understanding_provider']})" if data.get("understanding_provider") else "")
            + f"  |  Search query sent: \"{data['search_query_used']}\""
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Query understanding")
            st.json(data["analysis"])
        with col2:
            st.subheader("Adaptive weights used")
            st.json(data["weights"])

        st.subheader(f"Ranked sources (top {len(data['results'])})")
        for i, r in enumerate(data["results"], start=1):
            scores = r["scores"]
            with st.expander(
                f"{i}. {r['source_title'] or r['domain']}  —  final score: {scores['final']}"
            ):
                st.write(f"**URL:** {r['source_url']}")
                st.write(f"**Published:** {r['published_date'] or 'unknown'}  |  **Author:** {r['author'] or 'unknown'}")
                st.write(
                    f"**Scores** — semantic: `{scores['semantic']}`, "
                    f"freshness: `{scores['freshness']}`, "
                    f"authority: `{scores['authority']}`, "
                    f"credibility: `{scores['credibility']}`"
                )
                st.write(r["text"][:800] + ("..." if len(r["text"]) > 800 else ""))

        with st.expander("Pipeline stats / timings"):
            st.json(data["stats"])
