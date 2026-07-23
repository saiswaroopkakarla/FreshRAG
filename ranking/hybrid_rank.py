"""
Module 11 -- Hybrid Ranker.

Combines the four component scores using the (adaptively-generated)
weights into one final score per chunk:

    final = w_sem*semantic + w_fresh*freshness + w_auth*authority + w_cred*credibility

sorts chunks by final score descending, then applies a source-diversity
pass: since freshness/authority/credibility are per-document (identical
across every chunk of the same page) and only semantic relevance varies
chunk-to-chunk, a single strong page can otherwise supply most/all of
the top-k with its own chunks, crowding out every other source. This
caps how many chunks from the same URL can cluster at the top, without
discarding anything -- excess chunks are pushed later in the list
rather than dropped, so a caller taking `top_k` naturally gets a more
source-diverse result set.
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


def _diversify_by_source(sorted_chunks: list[ScoredChunk], max_per_source: int) -> list[ScoredChunk]:
    """Greedy pass: keep original (score-descending) relative order, but
    once a source URL has contributed `max_per_source` chunks, defer any
    further chunks from that same URL to the end of the list instead of
    letting them occupy the next-highest-scoring slots."""
    accepted = []
    deferred = []
    source_counts: dict[str, int] = {}

    for sc in sorted_chunks:
        url = sc.chunk.source_url
        count = source_counts.get(url, 0)
        if count < max_per_source:
            accepted.append(sc)
            source_counts[url] = count + 1
        else:
            deferred.append(sc)

    return accepted + deferred


def rank_chunks(
    scored_chunks: list[ScoredChunk],
    weights: RankingWeights,
    max_chunks_per_source: int | None = None,
) -> list[ScoredChunk]:
    for sc in scored_chunks:
        sc.final_score = compute_final_score(
            sc.semantic_score, sc.freshness_score, sc.authority_score, sc.credibility_score, weights
        )
    ranked = sorted(scored_chunks, key=lambda c: c.final_score, reverse=True)

    if max_chunks_per_source is not None:
        ranked = _diversify_by_source(ranked, max_chunks_per_source)

    return ranked
