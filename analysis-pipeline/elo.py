"""Surface-weighted ELO with a dynamic K-factor.

Keeps an overall rating per player plus one rating per surface, and blends
them at prediction time. Ratings update online, match by match, in
chronological order -- never feed it future matches.
"""

DEFAULT_RATING = 1500.0


def expected_score(r_a: float, r_b: float) -> float:
    """Probability A beats B given two ELO ratings."""
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def k_factor(matches_played: int) -> float:
    """Dynamic K: moves new players fast, veterans slowly (538 tennis form)."""
    return 250.0 / (matches_played + 5) ** 0.4


class EloModel:
    def __init__(self, surface_weight: float = 0.5):
        # weight on the surface-specific rating vs the overall rating
        self.surface_weight = surface_weight
        self.overall: dict[str, float] = {}
        self.surface: dict[tuple[str, str], float] = {}
        self.matches: dict[str, int] = {}

    def _overall(self, p: str) -> float:
        return self.overall.get(p, DEFAULT_RATING)

    def _surface(self, p: str, s: str) -> float:
        return self.surface.get((p, s), DEFAULT_RATING)

    def effective(self, p: str, s: str) -> float:
        """Blended rating actually used for prediction."""
        w = self.surface_weight
        return w * self._surface(p, s) + (1 - w) * self._overall(p)

    def predict(self, a: str, b: str, s: str) -> float:
        """P(a beats b) on surface s, from blended ratings."""
        return expected_score(self.effective(a, s), self.effective(b, s))

    def update(self, a: str, b: str, s: str, a_won: bool) -> None:
        """Update both chains after a result. Call in chronological order."""
        score_a = 1.0 if a_won else 0.0

        # overall chain
        exp_a = expected_score(self._overall(a), self._overall(b))
        k_a = k_factor(self.matches.get(a, 0))
        k_b = k_factor(self.matches.get(b, 0))
        self.overall[a] = self._overall(a) + k_a * (score_a - exp_a)
        self.overall[b] = self._overall(b) + k_b * ((1 - score_a) - (1 - exp_a))

        # surface chain
        exp_a_s = expected_score(self._surface(a, s), self._surface(b, s))
        self.surface[(a, s)] = self._surface(a, s) + k_a * (score_a - exp_a_s)
        self.surface[(b, s)] = self._surface(b, s) + k_b * ((1 - score_a) - (1 - exp_a_s))

        self.matches[a] = self.matches.get(a, 0) + 1
        self.matches[b] = self.matches.get(b, 0) + 1
