"""
Module 1/2 (v2) -- LLM-based Query Understanding.

This replaces the rule-based keyword-table analyzer as the PRIMARY path.
The table approach (processing/query_analyzer.py) has a structural
ceiling: it can only recognize domains/phrasings someone explicitly
anticipated, so it silently mis-fires on typos, novel topics, or
anything outside its ~6 buckets (e.g. "whats the recent update on
FIFA" got classified as `tech` and searched as something close to
"WhatsApp update" -- see conversation history).

An LLM has no such ceiling: it reasons about arbitrary topics using
general world knowledge, tolerates typos naturally, and can assign
sensible ranking weights to domains it's never been told about
explicitly (e.g. "cooking", "medical research", "video games").

Design:
  - Tries providers in this order: Groq (fast, genuinely free tier) ->
    DeepSeek (cheap, OpenAI-compatible) -> OpenAI -> Anthropic. Uses
    whichever key is configured; if none are, returns None immediately
    so the caller falls back to the rule-based analyzer -- the system
    NEVER hard-fails just because no LLM key is set.
  - The LLM is asked to return strict JSON: time-sensitivity, intent,
    a free-text topic label (not restricted to a fixed list), search
    keywords, a corrected/cleaned search-engine query, and -- this is
    the important part -- suggested ranking weights directly, reasoning
    about *this specific query* rather than looking up a bucket.
  - Any failure (no key, network error, malformed JSON, timeout) is
    caught and returns None. The pipeline always has a working fallback.
"""

import json
import logging
import re

import requests

from app.config import get_settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are the query-understanding stage of a retrieval-augmented \
generation system. Analyze the user's question and return ONLY a JSON object \
(no markdown fences, no commentary) with exactly these fields:

{
  "time_sensitive": true or false,
  "intent": one short label, e.g. "reason", "comparison", "definition", "current_status", "lookup", "other",
  "topic": a short free-text label for what the query is actually about (not restricted to any fixed list -- use your own judgment, e.g. "finance", "football/FIFA", "home cooking", "medical research", "video games"),
  "keywords": [3 to 6 core content words/phrases a search engine should use -- correct obvious typos, drop filler/question words],
  "search_query": a clean, well-formed search-engine query string (correct typos, remove filler words, keep it short),
  "weights": {
    "semantic": float,
    "freshness": float,
    "authority": float,
    "credibility": float
  }
}

Guidance for the weights (they should sum to roughly 1.0):
- Raise "freshness" when the answer meaningfully changes day to day or hour to hour (news, prices, scores, weather, ongoing events).
- Raise "semantic" when the query is about a stable fact, concept, or definition that doesn't change with time.
- Raise "authority" when getting it wrong has real consequences (medical, financial, legal, safety) or the topic is prone to misinformation.
- Raise "credibility" similarly when source transparency (byline, dates) matters more than usual.
Use your own judgment per-query -- do not default to a fixed template."""


def _build_user_prompt(query: str) -> str:
    return f'User query: "{query}"\n\nReturn only the JSON object.'


def _extract_json(raw_text: str) -> dict | None:
    """LLMs sometimes wrap JSON in markdown fences despite instructions
    not to -- strip those before parsing, and fall back to finding the
    outermost {...} block if there's stray text around it."""
    text = raw_text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _validate_and_normalize(data: dict, original_query: str) -> dict | None:
    try:
        weights = data.get("weights", {})
        w_sem = float(weights.get("semantic", 0.4))
        w_fresh = float(weights.get("freshness", 0.3))
        w_auth = float(weights.get("authority", 0.2))
        w_cred = float(weights.get("credibility", 0.1))
        total = w_sem + w_fresh + w_auth + w_cred
        if total <= 0:
            w_sem, w_fresh, w_auth, w_cred, total = 0.4, 0.3, 0.2, 0.1, 1.0

        keywords = data.get("keywords", [])
        if not isinstance(keywords, list) or not keywords:
            keywords = original_query.split()[:6]

        search_query = data.get("search_query") or " ".join(str(k) for k in keywords)

        return {
            "time_sensitive": bool(data.get("time_sensitive", False)),
            "intent": str(data.get("intent", "other"))[:40],
            "topic": str(data.get("topic", "general"))[:60],
            "keywords": [str(k) for k in keywords][:8],
            "search_query": str(search_query)[:200],
            "weights": {
                "semantic": w_sem / total,
                "freshness": w_fresh / total,
                "authority": w_auth / total,
                "credibility": w_cred / total,
            },
        }
    except (TypeError, ValueError) as exc:
        logger.warning("LLM query-understanding response failed validation: %s", exc)
        return None


def _call_groq(query: str, api_key: str) -> str:
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(query)},
            ],
            "temperature": 0.1,
            "max_tokens": 400,
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_deepseek(query: str, api_key: str) -> str:
    resp = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(query)},
            ],
            "temperature": 0.1,
            "max_tokens": 400,
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_openai(query: str, api_key: str) -> str:
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(query)},
            ],
            "temperature": 0.1,
            "max_tokens": 400,
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_anthropic(query: str, api_key: str) -> str:
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 400,
            "system": _SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": _build_user_prompt(query)}],
        },
        timeout=15,
    )
    resp.raise_for_status()
    blocks = resp.json().get("content", [])
    return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


# Priority order: fast free tier first, then cheap, then whatever else is configured.
_PROVIDERS = [
    ("groq", "groq_api_key", _call_groq),
    ("deepseek", "deepseek_api_key", _call_deepseek),
    ("openai", "openai_api_key", _call_openai),
    ("anthropic", "anthropic_api_key", _call_anthropic),
]


def llm_understand_query(query: str) -> dict | None:
    """
    Returns a dict with keys: time_sensitive, intent, topic, keywords,
    search_query, weights, provider -- or None if no key is configured
    or every configured provider failed. Callers MUST handle None by
    falling back to the rule-based analyzer.
    """
    settings = get_settings()

    for provider_name, key_attr, call_fn in _PROVIDERS:
        api_key = getattr(settings, key_attr)
        if not api_key:
            continue
        try:
            raw = call_fn(query, api_key)
            parsed = _extract_json(raw)
            if parsed is None:
                logger.warning("%s returned unparseable JSON: %r", provider_name, raw[:200])
                continue
            validated = _validate_and_normalize(parsed, query)
            if validated is None:
                continue
            validated["provider"] = provider_name
            return validated
        except Exception as exc:  # noqa: BLE001 -- any provider failure just moves to the next
            logger.warning("%s query-understanding call failed: %s", provider_name, exc)
            continue

    return None


def understand_query(query: str) -> dict:
    """
    Single entrypoint the pipeline calls. Returns a unified dict shape
    regardless of which path was used:

        {
          "query": str, "time_sensitive": bool, "intent": str,
          "domain": str, "keywords": [str], "search_query": str,
          "weights": {"semantic":.., "freshness":.., "authority":.., "credibility":..},
          "method": "llm" | "rule-based", "provider": str | None,
        }

    Mode is controlled by QUERY_UNDERSTANDING_MODE in .env:
      - "auto" (default): try LLM if any key is configured, else rule-based.
      - "llm": force LLM; raises RuntimeError if no key is configured or
        every provider fails (use this only if you want a hard failure
        instead of silent fallback, e.g. while benchmarking).
      - "rule-based": always use the keyword-table analyzer, even if an
        LLM key is set -- lets you A/B the two approaches for a thesis
        ablation without touching .env keys.
    """
    from processing.query_analyzer import analyze_query, clean_search_query
    from ranking.weight_generator import generate_weights

    settings = get_settings()
    mode = settings.query_understanding_mode

    def _rule_based_result() -> dict:
        analysis = analyze_query(query)
        weights = generate_weights(analysis)
        return {
            "query": query,
            "time_sensitive": analysis.time_sensitive,
            "intent": analysis.intent,
            "domain": analysis.domain,
            "keywords": analysis.keywords,
            "search_query": clean_search_query(analysis),
            "weights": weights.to_dict(),
            "method": "rule-based",
            "provider": None,
        }

    if mode == "rule-based":
        return _rule_based_result()

    llm_result = llm_understand_query(query)

    if llm_result is None:
        if mode == "llm":
            raise RuntimeError(
                "QUERY_UNDERSTANDING_MODE=llm but no LLM provider succeeded. "
                "Set GROQ_API_KEY, DEEPSEEK_API_KEY, OPENAI_API_KEY, or "
                "ANTHROPIC_API_KEY in .env, or switch mode to 'auto'."
            )
        logger.info("No LLM query-understanding available; falling back to rule-based analyzer.")
        return _rule_based_result()

    return {
        "query": query,
        "time_sensitive": llm_result["time_sensitive"],
        "intent": llm_result["intent"],
        "domain": llm_result["topic"],
        "keywords": llm_result["keywords"],
        "search_query": llm_result["search_query"],
        "weights": {k: round(v, 3) for k, v in llm_result["weights"].items()},
        "method": "llm",
        "provider": llm_result["provider"],
    }
