"""Hierarchical Markov serve model.

Tennis nests cleanly: one number per player -- their probability of winning a
point on serve in this matchup -- cascades up into a match win probability.

    serve-point prob  ->  game  ->  set (+ tiebreak)  ->  match
"""

from functools import lru_cache


def _clip(p: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return max(lo, min(hi, p))


def serve_point_prob(spw_server: float, rpw_returner: float,
                     tour_avg_spw: float) -> float:
    """Barnett-Clarke: server's point-win prob vs this specific returner.

    p = f_t + (server's serve edge) - (returner's return edge)
      = tour_avg + (spw_server - tour_avg) - (rpw_returner - (1 - tour_avg))
    """
    p = tour_avg_spw + (spw_server - tour_avg_spw) - (rpw_returner - (1 - tour_avg_spw))
    return _clip(p)


def game_win_prob(p: float) -> float:
    """P(server holds) given point-win prob p. Win 4 points, win by 2."""
    q = 1 - p
    # P(win from deuce): win next two, or split and return to deuce
    p_deuce = (p * p) / (p * p + q * q) if (p * p + q * q) > 0 else 0.5
    return (p ** 4                       # to love   (4-0)
            + 4 * p ** 4 * q             # to 15     (4-1)
            + 10 * p ** 4 * q ** 2       # to 30     (4-2)
            + 20 * p ** 3 * q ** 3 * p_deuce)  # reach deuce then win


def _tb_server_is_a(a_pts: int, b_pts: int) -> bool:
    """Tiebreak serve rotation (1-2-2-2...): A serves pt 1, then alternating pairs."""
    n = a_pts + b_pts
    if n == 0:
        return True
    pair = (n - 1) // 2
    return pair % 2 == 1  # pair 0 -> B, pair 1 -> A, ...


def tiebreak_win_prob(p_a: float, p_b: float) -> float:
    """P(A wins tiebreak): first to 7, win by 2, with serve rotation."""
    # Tail value once tied at >= 6 (deuce-style). Over any 2-point cycle at a
    # tie there is exactly one A-serve and one B-serve point, so the value is
    # phase-independent: it depends only on the two per-point A-win probs.
    a_on_serve, a_on_return = p_a, (1 - p_b)
    denom = a_on_serve * a_on_return + (1 - a_on_serve) * (1 - a_on_return)
    tail = (a_on_serve * a_on_return) / denom if denom > 0 else 0.5

    @lru_cache(maxsize=None)
    def f(a: int, b: int) -> float:
        if a == b and a >= 6:          # deuce tail (also bounds recursion)
            return tail
        if a >= 7 and a - b >= 2:
            return 1.0
        if b >= 7 and b - a >= 2:
            return 0.0
        p_pt = p_a if _tb_server_is_a(a, b) else (1 - p_b)
        return p_pt * f(a + 1, b) + (1 - p_pt) * f(a, b + 1)

    res = f(0, 0)
    f.cache_clear()
    return res


def set_win_prob(p_a: float, p_b: float) -> float:
    """P(A wins a set): first to 6, win by 2, tiebreak at 6-6.

    No closed form because serve alternates each game, so a small DP over
    (A games, B games, who serves). Averaged over who serves first (tiny effect).
    """
    g_a = game_win_prob(p_a)   # A holds
    g_b = game_win_prob(p_b)   # B holds  -> A breaks with prob (1 - g_b)
    tb = tiebreak_win_prob(p_a, p_b)

    @lru_cache(maxsize=None)
    def f(a: int, b: int, a_serves: bool) -> float:
        if a == 6 and b == 6:
            return tb
        if a >= 6 and a - b >= 2:
            return 1.0
        if b >= 6 and b - a >= 2:
            return 0.0
        p_win_game = g_a if a_serves else (1 - g_b)
        return (p_win_game * f(a + 1, b, not a_serves)
                + (1 - p_win_game) * f(a, b + 1, not a_serves))

    res = 0.5 * f(0, 0, True) + 0.5 * f(0, 0, False)
    f.cache_clear()
    return res


def match_win_prob_from_set(s: float, best_of: int = 3) -> float:
    """P(A wins match) from per-set prob s, treating sets as independent."""
    if best_of == 3:
        return s ** 2 * (3 - 2 * s)
    if best_of == 5:
        return s ** 3 * (6 * s ** 2 - 15 * s + 10)
    raise ValueError("best_of must be 3 or 5")


def markov_match_prob(spw_a: float, rpw_a: float, spw_b: float, rpw_b: float,
                      tour_avg_spw: float, best_of: int = 3) -> float:
    """End-to-end: serve/return rates -> match win probability for A."""
    p_a = serve_point_prob(spw_a, rpw_b, tour_avg_spw)  # A serving vs B returning
    p_b = serve_point_prob(spw_b, rpw_a, tour_avg_spw)  # B serving vs A returning
    s = set_win_prob(p_a, p_b)
    return match_win_prob_from_set(s, best_of)
