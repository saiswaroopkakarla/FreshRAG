"""
FreshRAG -- FastAPI backend entrypoint.

Run with:
    uvicorn app.main:app --reload

Then open http://localhost:8000/docs for interactive Swagger UI.

Why FastAPI instead of Streamlit for the backend: this project is
architected as a real API-first service (so it can later serve a web
frontend, the Streamlit demo UI, or even a mobile app) rather than a
single throwaway script. Streamlit is used separately (streamlit_app.py)
purely as a thin demo client that calls this API.
"""

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import get_settings
from app.logging_config import setup_logging
from app.pipeline import run_pipeline
from processing.llm_understanding import understand_query
from utils.exceptions import FreshRAGError

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FreshRAG API",
    description=(
        "Adaptive Multi-Source Temporal-Aware Hybrid RAG. Ranks retrieved "
        "web content by semantic relevance, freshness, authority, and "
        "credibility, with adaptively-generated weights per query."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Natural-language question")
    top_k: int | None = Field(None, ge=1, le=30, description="Number of ranked sources to return")


class AnalyzeRequest(BaseModel):
    query: str = Field(..., min_length=2)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/config")
def config():
    settings = get_settings()
    return {
        "query_understanding_mode": settings.query_understanding_mode,
        "embedding_mode": settings.embedding_mode,
        "freshness_decay": settings.freshness_decay,
        "top_k_results": settings.top_k_results,
        "max_urls_to_fetch": settings.max_urls_to_fetch,
        "has_groq_key": bool(settings.groq_api_key),
        "has_deepseek_key": bool(settings.deepseek_api_key),
        "has_openai_key": bool(settings.openai_api_key),
        "has_anthropic_key": bool(settings.anthropic_api_key),
        "has_newsapi_key": bool(settings.newsapi_key),
        "has_tavily_key": bool(settings.tavily_api_key),
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """Debug endpoint: run only Stage 1 (query understanding: LLM if a
    key is configured, else the rule-based fallback) without doing any
    retrieval. Useful for quickly checking how a query gets classified
    and weighted before spending time on the full pipeline."""
    try:
        return understand_query(req.query)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/query")
def query(req: QueryRequest):
    """Full pipeline: retrieve, score, rank, and generate an answer."""
    try:
        result = run_pipeline(req.query, top_k=req.top_k)
        return result
    except FreshRAGError as exc:
        logger.warning("Pipeline error for query '%s': %s", req.query, exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error for query '%s'", req.query)
        raise HTTPException(status_code=500, detail="Internal server error") from exc
