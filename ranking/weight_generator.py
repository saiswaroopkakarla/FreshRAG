"""
Module 11 (partial) -- Adaptive Weight Generator.

*** This is the main research contribution of the project. ***

Instead of a single fixed formula like:

    final = 0.45*semantic + 0.30*freshness + 0.15*authority + 0.10*credibility

for every query, the weights themselves are adapted based on the query's
domain and time-sensitivity (from processing/query_analyzer.py). E.g.:

  - "Why is Apple stock crashing today?" (finance, time-sensitive)
    -> freshness should dominate; a semantically-similar-but-3-year-old
       article about Apple earnings is actively unhelpful.

  - "What is the Pythagorean theorem?" (general, not time-sensitive)
    -> freshness is irrelevant; semantic relevance should dominate.

This file is intentionally simple/tunable rule table for now (Phase 1).
The natural "v2" research extension discussed in the design notes is to
replace this table with a learned model (e.g. a small logistic
regression / learning-to-rank model trained on click/relevance
feedback) -- the `generate_weights` function signature is written so
that swap is a drop-in replacement later.
"""

from dataclasses import dataclass

from app.config import get_settings
from processing.query_analyzer import QueryAnalysis


@dataclass
class RankingWeights:
    semantic: float
    freshness: float
    authority: float
    credibility: float

    def normalized(self) -> "RankingWeights":
        total = self.semantic + self.freshness + self.authority + self.credibility
        if total <= 0:
            return RankingWeights(0.25, 0.25, 0.25, 0.25)
        return RankingWeights(
            semantic=self.semantic / total,
            freshness=self.freshness / total,
            authority=self.authority / total,
            credibility=self.credibility / total,
        )

    def to_dict(self) -> dict:
        w = self.normalized()
        return {
            "semantic": round(w.semantic, 3),
            "freshness": round(w.freshness, 3),
            "authority": round(w.authority, 3),
            "credibility": round(w.credibility, 3),
        }


# Domain -> base weight profile (before the time-sensitivity adjustment).
# These are starting points for experimentation / the ablation study,
# not claimed-optimal values.
_DOMAIN_PROFILES = {
    "finance": RankingWeights(semantic=0.30, freshness=0.45, authority=0.15, credibility=0.10),
    "sports": RankingWeights(semantic=0.30, freshness=0.45, authority=0.15, credibility=0.10),
    "news": RankingWeights(semantic=0.30, freshness=0.40, authority=0.20, credibility=0.10),
    "weather": RankingWeights(semantic=0.20, freshness=0.60, authority=0.10, credibility=0.10),
    "tech": RankingWeights(semantic=0.45, freshness=0.30, authority=0.15, credibility=0.10),
    "general": RankingWeights(semantic=0.55, freshness=0.15, authority=0.20, credibility=0.10),
}


def generate_weights(analysis: QueryAnalysis) -> RankingWeights:
    """Adaptive weight generation. This is the function to replace with a
    learned model for the "v2" research extension."""
    settings = get_settings()
    base = _DOMAIN_PROFILES.get(
        analysis.domain,
        RankingWeights(
            settings.default_weight_semantic,
            settings.default_weight_freshness,
            settings.default_weight_authority,
            settings.default_weight_credibility,
        ),
    )

    if not analysis.time_sensitive:
        # If the query isn't time-sensitive, pull weight back from
        # freshness towards semantic relevance regardless of domain.
        shift = base.freshness * 0.6
        base = RankingWeights(
            semantic=base.semantic + shift,
            freshness=base.freshness - shift,
            authority=base.authority,
            credibility=base.credibility,
        )

    return base.normalized()
