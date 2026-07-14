"""
Module 6 -- Semantic Relevance Scoring.

Thin wrapper around embedding.embedder so the ranking package has a
consistent "one module per score" structure, mirroring the design
doc's module list (relevance / freshness / authority / credibility).
"""

from embedding.embedder import compute_relevance_scores

__all__ = ["compute_relevance_scores"]
