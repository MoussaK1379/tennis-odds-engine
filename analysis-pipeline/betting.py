"""Turning a calibrated probability + bookmaker odds into a staking decision."""


def implied_prob(decimal_odds: float) -> float:
    """Raw implied probability from decimal odds (includes the vig)."""
    return 1.0 / decimal_odds


def devig_two_way(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Strip the bookmaker margin from a two-way market (proportional method).

    Returns fair probabilities for A and B that sum to 1. Simple baseline;
    Shin / logarithmic methods are better upgrades later.
    """
    q_a, q_b = implied_prob(odds_a), implied_prob(odds_b)
    total = q_a + q_b  # > 1; the excess is the overround
    return q_a / total, q_b / total


def overround(odds_a: float, odds_b: float) -> float:
    """Bookmaker margin on a two-way market (e.g. 0.05 == 5%)."""
    return implied_prob(odds_a) + implied_prob(odds_b) - 1.0


def expected_value(p: float, decimal_odds: float) -> float:
    """EV of a 1-unit bet. Positive => bet has edge. Edge == this value."""
    return p * decimal_odds - 1.0


def kelly_fraction(p: float, decimal_odds: float, fraction: float = 0.25) -> float:
    """Fraction of bankroll to stake (fractional Kelly).

    Full Kelly assumes your probability is exactly right; it isn't, so scale
    down. Returns 0 when there's no edge (never stake into negative EV).
    """
    b = decimal_odds - 1.0
    q = 1.0 - p
    f_star = (b * p - q) / b
    return max(0.0, fraction * f_star)
