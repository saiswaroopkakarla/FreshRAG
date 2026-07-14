"""
Central configuration for FreshRAG.

Everything is loaded from environment variables (via a `.env` file). This
module is the single source of truth for tunable parameters, so ranking
weights, decay functions, and provider choices can be changed without
touching pipeline code -- important for running ablation experiments for
the research write-up (e.g. "exponential vs linear freshness decay").
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Optional API keys (system works with none of these set) ---
    newsapi_key: str = ""
    tavily_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    groq_api_key: str = ""
    deepseek_api_key: str = ""

    # --- Query understanding ---
    # auto: use an LLM if any key above is set, else the rule-based
    #       analyzer (this is what most people want).
    # llm: force LLM-based understanding; raises if no key is configured.
    # rule-based: force the keyword-table analyzer even if a key is set
    #             (useful for an ablation: "rule-based vs LLM query
    #             understanding" comparison in your write-up).
    query_understanding_mode: Literal["auto", "llm", "rule-based"] = "auto"

    # --- Embeddings ---
    embedding_mode: Literal["tfidf", "sentence-transformers"] = "tfidf"

    # --- Freshness scoring ---
    freshness_decay: Literal["exponential", "linear", "logistic", "piecewise"] = "exponential"
    # Half-life-ish constant for exponential decay (days). Larger = slower decay.
    freshness_lambda_days: float = 3.0
    # Max age (days) considered for linear decay before score floors at 0.
    freshness_max_age_days: float = 365.0

    # --- Retrieval tuning ---
    top_k_results: int = 8
    max_urls_to_fetch: int = 10
    chunk_size_words: int = 220
    chunk_overlap_words: int = 40

    # --- Default hybrid ranking weights (used when adaptive weighting
    #     has no stronger opinion). Must sum to ~1.0. ---
    default_weight_semantic: float = 0.45
    default_weight_freshness: float = 0.30
    default_weight_authority: float = 0.15
    default_weight_credibility: float = 0.10

    # --- Frontend ---
    backend_url: str = "http://localhost:8000"

    # --- App ---
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor -- avoids re-reading .env on every call."""
    return Settings()
