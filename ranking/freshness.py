"""
Module 7 -- Freshness Scoring. This is the project's main research
scoring component.

Implements four decay functions, selectable via FRESHNESS_DECAY in
.env, so they can be compared head-to-head as an ablation study:

  1. Linear:      F = max(0, 1 - age/max_age)
  2. Exponential: F = e^(-lambda * age)         [most common in IR literature]
  3. Logistic:    F = 1 / (1 + e^(k*(age - midpoint)))   -- smooth "cliff"
  4. Piecewise:   flat near 1.0 for very fresh content, then decays

All functions take `age_days` (float, >= 0) and return a score in [0, 1].
A document with no extractable published date gets a neutral fallback
score (see `score_document`) rather than being penalized to zero --
we don't want to punish a page just because it lacked <meta> tags.
"""

import math
from datetime import datetime, timezone

from app.config import get_settings

NEUTRAL_UNKNOWN_DATE_SCORE = 0.5

# math.exp() raises OverflowError for inputs beyond roughly 709 (the
# limit of a 64-bit float). Any exponent magnitude at or above this is
# already so extreme that the result would be indistinguishable from
# 0.0 or 1.0 anyway, so clamping the exponent before calling exp() is
# mathematically safe -- it changes no real output, it just prevents a
# crash on inputs that would have produced ~0 or ~1 regardless.
_MAX_EXP_ARG = 700.0


def _safe_exp(x: float) -> float:
    """exp() that saturates instead of raising OverflowError for very
    large magnitude inputs (in either direction)."""
    if x > _MAX_EXP_ARG:
        x = _MAX_EXP_ARG
    elif x < -_MAX_EXP_ARG:
        x = -_MAX_EXP_ARG
    return math.exp(x)


def linear_decay(age_days: float, max_age_days: float) -> float:
    if max_age_days <= 0:
        return 0.0
    return max(0.0, 1.0 - (age_days / max_age_days))


def exponential_decay(age_days: float, lambda_days: float) -> float:
    if lambda_days <= 0:
        lambda_days = 1.0
    rate = 1.0 / lambda_days
    return _safe_exp(-rate * age_days)


def logistic_decay(age_days: float, midpoint_days: float = 7.0, steepness: float = 0.5) -> float:
    return 1.0 / (1.0 + _safe_exp(steepness * (age_days - midpoint_days)))


def piecewise_decay(age_days: float) -> float:
    """Flat near-1.0 for the first day, fast decay through the first
    week, then a slow long tail -- mirrors how a human would judge
    "is this news fresh?"."""
    if age_days <= 1:
        return 1.0
    if age_days <= 7:
        # Linearly drop from 1.0 -> 0.6 over days 1-7.
        return 1.0 - 0.4 * ((age_days - 1) / 6)
    if age_days <= 30:
        # Linearly drop from 0.6 -> 0.2 over days 7-30.
        return 0.6 - 0.4 * ((age_days - 7) / 23)
    # Long tail: slow exponential decay from 0.2 downward.
    return 0.2 * _safe_exp(-(age_days - 30) / 180)


def compute_freshness_score(age_days: float, decay_fn: str | None = None) -> float:
    settings = get_settings()
    decay_fn = decay_fn or settings.freshness_decay

    if decay_fn == "linear":
        score = linear_decay(age_days, settings.freshness_max_age_days)
    elif decay_fn == "logistic":
        score = logistic_decay(age_days)
    elif decay_fn == "piecewise":
        score = piecewise_decay(age_days)
    else:  # exponential (default)
        score = exponential_decay(age_days, settings.freshness_lambda_days)

    return max(0.0, min(1.0, score))


def score_document(published_date, decay_fn: str | None = None) -> float:
    """
    published_date: datetime | None. Returns freshness score in [0, 1].
    """
    if published_date is None:
        return NEUTRAL_UNKNOWN_DATE_SCORE

    now = datetime.now(timezone.utc)
    if published_date.tzinfo is None:
        published_date = published_date.replace(tzinfo=timezone.utc)

    age_days = max(0.0, (now - published_date).total_seconds() / 86400.0)
    return compute_freshness_score(age_days, decay_fn)
