"""
Module 10 -- Credibility Scoring.

Distinct from Authority: Authority asks "how reputable is this
publisher in general?", Credibility asks "does *this specific page*
show the transparency signals of a trustworthy piece of content?" --
i.e. does it disclose an author, a clear publish date, and use HTTPS.

This is a lightweight heuristic (each signal contributes a fixed
amount), not a claim of solving misinformation detection -- it's a
reasonable, explainable proxy for the ranking pipeline.
"""


def score_credibility(author: str, has_published_date: bool, url: str) -> float:
    score = 0.4  # baseline

    if author and author.strip():
        score += 0.25

    if has_published_date:
        score += 0.25

    if url.lower().startswith("https://"):
        score += 0.10

    return max(0.0, min(1.0, score))
