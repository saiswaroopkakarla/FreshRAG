"""
Module 12/Evaluation -- Ranking quality metrics.

Standard IR metrics (Precision@k, Recall@k, nDCG@k) require a labeled
ground-truth relevance set, which you'll build as part of the thesis
evaluation (e.g. manually judging relevance for a query set). The
functions here are ready to consume that once you have it.

`freshness_satisfaction` is the project's own proposed metric (noted in
the design discussion) -- it doesn't need external labels, so it works
out of the box on any live query and is a good "quick sanity check"
number to show during a demo.
"""

import math


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def dcg_at_k(relevance_scores: list[float], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k], start=1):
        dcg += (2**rel - 1) / math.log2(i + 1)
    return dcg


def ndcg_at_k(relevance_scores: list[float], k: int) -> float:
    """relevance_scores: graded relevance (e.g. 0-3) for the retrieved
    docs, in the order they were retrieved/ranked."""
    ideal = sorted(relevance_scores, reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(relevance_scores, k) / ideal_dcg


def freshness_satisfaction(freshness_scores: list[float], threshold: float = 0.5) -> float:
    """
    Proposed custom metric: the fraction of top-ranked results that meet
    a 'fresh enough' bar. Useful as a single number to track when
    comparing decay functions (linear vs exponential vs logistic vs
    piecewise) for a given query set -- no ground-truth labels required.
    """
    if not freshness_scores:
        return 0.0
    satisfied = sum(1 for f in freshness_scores if f >= threshold)
    return satisfied / len(freshness_scores)
