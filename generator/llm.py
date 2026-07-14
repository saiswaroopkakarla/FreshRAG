"""
Module 12 -- Answer Generation.

Pluggable across four providers, tried in this order: Anthropic -> OpenAI
-> Groq -> DeepSeek -> built-in extractive summarizer. Anthropic/OpenAI
are tried first only because if you've deliberately paid for one of
those keys you likely want its (typically higher) synthesis quality;
Groq and DeepSeek are free/cheap options that still give you a real
synthesized, cited answer with zero cost. If none are configured, the
extractive fallback stitches the top-ranked chunks together instead --
no API key, no cost, still runs instantly.

Note: this provider chain is independent from query understanding
(processing/llm_understanding.py). Setting one key (e.g. GROQ_API_KEY)
gets used for BOTH stages automatically -- no extra config needed.
"""

import logging

import requests

from app.config import get_settings
from embedding.vector_store import ScoredChunk

logger = logging.getLogger(__name__)

_GENERATION_INSTRUCTIONS = (
    "You are answering a user's question using ONLY the sources below, "
    "which are ranked by a hybrid relevance+freshness+authority score. "
    "Prioritize the freshest, most relevant sources. Cite sources as "
    "[Source N]. If sources conflict, prefer the more recent one and say so."
)


def _build_context(chunks: list[ScoredChunk]) -> str:
    parts = []
    for i, sc in enumerate(chunks, start=1):
        date_str = sc.chunk.published_date.strftime("%Y-%m-%d") if sc.chunk.published_date else "unknown date"
        parts.append(
            f"[Source {i}] ({sc.chunk.source_title or sc.chunk.domain}, {date_str})\n{sc.chunk.text}"
        )
    return "\n\n".join(parts)


def _build_prompt(query: str, context: str) -> str:
    return f"{_GENERATION_INSTRUCTIONS}\n\nSources:\n{context}\n\nQuestion: {query}\n\nAnswer:"


def _extractive_fallback(query: str, chunks: list[ScoredChunk]) -> str:
    """No-LLM-key fallback: stitches together the top-ranked chunks into
    a readable, cited answer. Deterministic and free."""
    if not chunks:
        return "I couldn't find any usable content to answer this query."

    lines = [f"Based on the top-ranked, freshest sources for: \"{query}\"\n"]
    for i, sc in enumerate(chunks, start=1):
        date_str = sc.chunk.published_date.strftime("%Y-%m-%d") if sc.chunk.published_date else "date unknown"
        snippet = sc.chunk.text[:400].rsplit(" ", 1)[0]
        lines.append(f"{i}. ({date_str}, {sc.chunk.domain}) {snippet}...")
    lines.append(
        "\n[Note: this is an extractive summary. Set GROQ_API_KEY (free), "
        "DEEPSEEK_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY in .env "
        "for a synthesized LLM answer instead.]"
    )
    return "\n".join(lines)


def _generate_with_anthropic(query: str, context: str, api_key: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        messages=[{"role": "user", "content": _build_prompt(query, context)}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def _generate_with_openai(query: str, context: str, api_key: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": _build_prompt(query, context)}],
        max_tokens=600,
    )
    return response.choices[0].message.content


def _generate_with_groq(query: str, context: str, api_key: str) -> str:
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            # A larger model than the one used for query understanding --
            # synthesis quality matters more here than raw speed, and
            # Groq's free tier still serves this fast.
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": _build_prompt(query, context)}],
            "temperature": 0.3,
            "max_tokens": 700,
        },
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _generate_with_deepseek(query: str, context: str, api_key: str) -> str:
    resp = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": _build_prompt(query, context)}],
            "temperature": 0.3,
            "max_tokens": 700,
        },
        timeout=25,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# (provider_name, settings_attr, call_fn) -- tried in order, first
# configured + successful one wins.
_PROVIDERS = [
    ("anthropic", "anthropic_api_key", _generate_with_anthropic),
    ("openai", "openai_api_key", _generate_with_openai),
    ("groq", "groq_api_key", _generate_with_groq),
    ("deepseek", "deepseek_api_key", _generate_with_deepseek),
]


def generate_answer(query: str, chunks: list[ScoredChunk]) -> dict:
    settings = get_settings()
    context = _build_context(chunks)

    for provider_name, key_attr, call_fn in _PROVIDERS:
        api_key = getattr(settings, key_attr)
        if not api_key:
            continue
        try:
            answer = call_fn(query, context, api_key)
            if answer and answer.strip():
                return {"answer": answer, "provider": provider_name}
            logger.warning("%s returned an empty answer; trying next option.", provider_name)
        except Exception as exc:  # noqa: BLE001 -- any provider failure just moves to the next
            logger.warning("%s generation failed (%s); trying next option.", provider_name, exc)

    return {"answer": _extractive_fallback(query, chunks), "provider": "extractive-fallback"}
