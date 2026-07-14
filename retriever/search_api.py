"""
Module 2/4 -- Multi-source web retrieval.

Design decision from the project discussion: don't lock into a single
paid search provider. FreshRAG works out of the box with DuckDuckGo
(free, no API key). If NEWSAPI_KEY / TAVILY_API_KEY are set in .env,
their results are merged in too -- this is the "Adaptive Multi-Source
Retrieval" idea from the research notes: use whichever sources are
available and let the ranking stage decide what's actually useful.

Every result is normalized to the same shape:
    {"url": str, "title": str, "snippet": str, "source": str}

Known limitation: DuckDuckGo's free search (via `ddgs`) is a scraping
wrapper, not an official API -- it can rate-limit or transiently fail,
especially under repeated requests. `_search_duckduckgo` retries with
backoff and tries multiple backends before giving up. If you hit
persistent failures, add a free NEWSAPI_KEY or TAVILY_API_KEY to .env
as a more reliable path -- both have generous free tiers.
"""

import logging
import time

import requests

from app.config import get_settings
from utils.exceptions import SearchProviderError

logger = logging.getLogger(__name__)

_DDG_BACKENDS = ["auto", "html", "lite"]


def _search_duckduckgo(query: str, max_results: int, retries: int = 2) -> list[dict]:
    try:
        from ddgs import DDGS
    except ImportError:
        logger.warning("ddgs package not installed; skipping DuckDuckGo search.")
        return []

    for attempt in range(retries + 1):
        for backend in _DDG_BACKENDS:
            try:
                results = []
                with DDGS() as ddgs:
                    for r in ddgs.text(query, max_results=max_results, backend=backend):
                        results.append(
                            {
                                "url": r.get("href") or r.get("link", ""),
                                "title": r.get("title", ""),
                                "snippet": r.get("body", ""),
                                "source": "duckduckgo",
                            }
                        )
                if results:
                    return results
            except Exception as exc:  # noqa: BLE001 - external service, keep pipeline alive
                logger.warning(
                    "DuckDuckGo search failed (backend=%s, attempt=%d/%d): %s",
                    backend, attempt + 1, retries + 1, exc,
                )
        if attempt < retries:
            wait = 1.5 * (attempt + 1)
            logger.info("Retrying DuckDuckGo search in %.1fs ...", wait)
            time.sleep(wait)

    logger.warning(
        "DuckDuckGo search exhausted all backends/retries. If this keeps "
        "happening, add a free NEWSAPI_KEY or TAVILY_API_KEY to .env."
    )
    return []


def _search_newsapi(query: str, max_results: int) -> list[dict]:
    settings = get_settings()
    if not settings.newsapi_key:
        return []

    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "sortBy": "publishedAt",
                "pageSize": max_results,
                "language": "en",
                "apiKey": settings.newsapi_key,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for article in data.get("articles", []):
            results.append(
                {
                    "url": article.get("url", ""),
                    "title": article.get("title", ""),
                    "snippet": article.get("description") or "",
                    "source": "newsapi",
                    "published_at": article.get("publishedAt"),
                }
            )
        return results
    except Exception as exc:  # noqa: BLE001
        logger.warning("NewsAPI search failed: %s", exc)
        return []


def _search_tavily(query: str, max_results: int) -> list[dict]:
    settings = get_settings()
    if not settings.tavily_api_key:
        return []

    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": settings.tavily_api_key,
                "query": query,
                "max_results": max_results,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("results", []):
            results.append(
                {
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "snippet": item.get("content", ""),
                    "source": "tavily",
                }
            )
        return results
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tavily search failed: %s", exc)
        return []


def multi_source_search(query: str, domain: str, time_sensitive: bool) -> list[dict]:
    """
    Adaptive multi-source search: always tries DuckDuckGo (free baseline),
    and adds NewsAPI when the query is news/finance-flavored and time
    sensitive (NewsAPI is date-sorted, which is exactly what we want
    there), plus Tavily if configured, for extra recall.
    """
    settings = get_settings()
    max_results = settings.max_urls_to_fetch

    all_results: list[dict] = []
    all_results.extend(_search_duckduckgo(query, max_results))

    if time_sensitive and domain in {"finance", "news", "sports"}:
        all_results.extend(_search_newsapi(query, max_results // 2))

    all_results.extend(_search_tavily(query, max_results // 2))

    if not all_results:
        raise SearchProviderError(
            "All search providers returned zero results. This usually means "
            "either (a) no internet connection, or (b) DuckDuckGo's free "
            "search is temporarily rate-limiting this IP (common -- it's a "
            "scraping-based free tier, not an official API). Wait a minute "
            "and retry, or add a free NEWSAPI_KEY / TAVILY_API_KEY to .env "
            "for a more reliable path."
        )

    # De-duplicate by URL while preserving first-seen order/source priority.
    seen_urls = set()
    deduped = []
    for r in all_results:
        url = r.get("url", "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        deduped.append(r)

    return deduped[: settings.max_urls_to_fetch]
