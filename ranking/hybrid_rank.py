"""
Module 11 -- Hybrid Ranker.

Combines the four component scores using the (adaptively-generated)
weights into one final score per chunk:

    final = w_sem*semantic + w_fresh*freshness + w_auth*authority + w_cred*credibility

and returns chunks sorted by final score descending.
"""

from embedding.vector_store import ScoredChunk
from ranking.weight_generator import RankingWeights


def compute_final_score(
    semantic: float,
    freshness: float,
    authority: float,
    credibility: float,
    weights: RankingWeights,
) -> float:
    w = weights.normalized()
    return (
        w.semantic * semantic
        + w.freshness * freshness
        + w.authority * authority
        + w.credibility * credibility
    )


def rank_chunks(scored_chunks: list[ScoredChunk], weights: RankingWeights) -> list[ScoredChunk]:
    for sc in scored_chunks:
        sc.final_score = compute_final_score(
            sc.semantic_score, sc.freshness_score, sc.authority_score, sc.credibility_score, weights
        )
    return sorted(scored_chunks, key=lambda c: c.final_score, reverse=True)
