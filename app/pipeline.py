"""
End-to-end FreshRAG pipeline orchestrator.

This is the "conductor" that runs the 12-module flow described in the
project design:

  Query Understanding -> Multi-Source Search -> Web Fetch -> Metadata
  Extraction -> Cleaning -> Chunking -> Relevance/Freshness/Authority/
  Credibility Scoring -> Adaptive Weight Generation -> Hybrid Ranking
  -> Answer Generation

Each step is intentionally a thin call into its own module so the
pipeline stays readable and each stage can be independently unit
tested / swapped out.

Query understanding (Stage 1) tries an LLM first (Groq/DeepSeek/OpenAI/
Anthropic -- whichever key is configured) and falls back automatically
to the rule-based keyword-table analyzer if no key is set or the LLM
call fails. See processing/llm_understanding.py for why: a fixed
keyword table can't cover open-domain queries or typos, but the system
still needs to work with zero API keys.
"""

import logging
import time

from app.config import get_settings
from embedding.embedder import compute_relevance_scores
from embedding.vector_store import ScoredChunk, SessionStore
from generator.llm import generate_answer
from processing.chunker import build_chunks_for_document
from processing.cleaner import extract_main_text, extract_title
from processing.llm_understanding import understand_query
from processing.metadata import extract_author, extract_domain, extract_published_date
from ranking.authority import score_authority
from ranking.credibility import score_credibility
from ranking.freshness import score_document as score_freshness
from ranking.hybrid_rank import rank_chunks
from ranking.weight_generator import RankingWeights
from retriever.search_api import multi_source_search
from retriever.web_fetcher import fetch_html
from utils.exceptions import FetchError, NoContentRetrievedError

logger = logging.getLogger(__name__)


def run_pipeline(query: str, top_k: int | None = None) -> dict:
    t0 = time.time()
    timings = {}

    # --- Stage 1: Query Understanding (LLM if available, else rule-based) ---
    step_t = time.time()
    understanding = understand_query(query)
    timings["query_understanding_s"] = round(time.time() - step_t, 3)
    logger.info(
        "Query understanding via %s%s: %s",
        understanding["method"],
        f" ({understanding['provider']})" if understanding.get("provider") else "",
        {k: v for k, v in understanding.items() if k != "weights"},
    )
    logger.info("Weights: %s", understanding["weights"])

    weights = RankingWeights(**understanding["weights"])
    search_query = understanding["search_query"]

    # --- Stage 2: Multi-source search ---
    step_t = time.time()
    logger.info("Search query sent to providers: '%s' (raw query was: '%s')", search_query, query)
    search_results = multi_source_search(
        search_query, understanding["domain"], understanding["time_sensitive"]
    )
    timings["search_s"] = round(time.time() - step_t, 3)
    logger.info("Retrieved %d candidate URLs", len(search_results))

    # --- Module 3/5/4: Fetch + clean + metadata + chunk ---
    step_t = time.time()
    store = SessionStore()
    fetched_count = 0
    for result in search_results:
        url = result["url"]
        try:
            html = fetch_html(url)
        except FetchError as exc:
            logger.debug("Skipping %s: %s", url, exc)
            continue

        text = extract_main_text(html)
        if len(text.split()) < 30:
            continue  # too little content to be useful

        title = extract_title(html) or result.get("title", "")
        published_date = extract_published_date(html)
        author = extract_author(html)
        domain = extract_domain(url)

        chunks = build_chunks_for_document(
            text=text,
            source_url=url,
            source_title=title,
            published_date=published_date,
            author=author,
            domain=domain,
        )
        for chunk in chunks:
            store.add(ScoredChunk(chunk=chunk))
        fetched_count += 1

    timings["fetch_and_chunk_s"] = round(time.time() - step_t, 3)
    logger.info("Successfully fetched+chunked %d/%d pages -> %d chunks", fetched_count, len(search_results), len(store.chunks))

    if not store.chunks:
        raise NoContentRetrievedError(
            "Could not extract usable content from any search result for this query."
        )

    # --- Module 6-10: Scoring ---
    step_t = time.time()
    chunk_texts = [sc.chunk.text for sc in store.chunks]
    relevance_scores = compute_relevance_scores(query, chunk_texts)

    for sc, rel_score in zip(store.chunks, relevance_scores):
        sc.semantic_score = rel_score
        sc.freshness_score = score_freshness(sc.chunk.published_date)
        sc.authority_score = score_authority(sc.chunk.domain)
        sc.credibility_score = score_credibility(
            sc.chunk.author, sc.chunk.published_date is not None, sc.chunk.source_url
        )
    timings["scoring_s"] = round(time.time() - step_t, 3)

    # --- Module 11: Hybrid ranking ---
    step_t = time.time()
    ranked = rank_chunks(store.chunks, weights, max_chunks_per_source=get_settings().max_chunks_per_source)
    k = top_k or 8
    top_chunks = ranked[:k]
    timings["ranking_s"] = round(time.time() - step_t, 3)

    # --- Module 12: Answer generation ---
    step_t = time.time()
    generation = generate_answer(query, top_chunks)
    timings["generation_s"] = round(time.time() - step_t, 3)

    timings["total_s"] = round(time.time() - t0, 3)

    return {
        "query": query,
        "search_query_used": search_query,
        "understanding_method": understanding["method"],
        "understanding_provider": understanding.get("provider"),
        "analysis": {
            "domain": understanding["domain"],
            "time_sensitive": understanding["time_sensitive"],
            "intent": understanding["intent"],
            "keywords": understanding["keywords"],
        },
        "weights": understanding["weights"],
        "answer": generation["answer"],
        "generation_provider": generation["provider"],
        "results": [sc.to_dict() for sc in top_chunks],
        "stats": {
            "urls_retrieved": len(search_results),
            "pages_fetched": fetched_count,
            "chunks_scored": len(store.chunks),
            "timings": timings,
        },
    }
