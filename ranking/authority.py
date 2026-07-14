"""
Module 9 -- Authority Scoring.

Simple, transparent heuristic: a curated tier list of well-known
reputable domains per category, plus generic boosts for .gov/.edu and
a small penalty for very low-signal domains. This is intentionally
simple (a lookup table) so it's easy to explain/defend in a thesis --
"here is exactly why source X scored higher than source Y" -- rather
than a black box.

Extension idea noted in the design discussion: replace/augment this
with a real domain-authority API (e.g. Moz/Ahrefs) if budget allows.
"""

_TIER_1_DOMAINS = {
    "reuters.com", "bloomberg.com", "apnews.com", "bbc.com", "nytimes.com",
    "wsj.com", "ft.com", "npr.org", "economist.com", "nature.com",
    "sec.gov", "who.int", "un.org", "espn.com", "cnbc.com",
}

_TIER_2_DOMAINS = {
    "cnn.com", "theguardian.com", "forbes.com", "techcrunch.com",
    "theverge.com", "wired.com", "arstechnica.com", "investing.com",
    "marketwatch.com", "yahoo.com", "coindesk.com", "espncricinfo.com",
}


def score_authority(domain: str) -> float:
    if not domain:
        return 0.3

    domain = domain.lower()

    if domain in _TIER_1_DOMAINS:
        return 1.0
    if domain in _TIER_2_DOMAINS:
        return 0.75

    if domain.endswith(".gov") or domain.endswith(".gov.in"):
        return 0.95
    if domain.endswith(".edu"):
        return 0.85
    if domain.endswith(".org"):
        return 0.6

    # Generic/unknown commercial domain.
    return 0.45
