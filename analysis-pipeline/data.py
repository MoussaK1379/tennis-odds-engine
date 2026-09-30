"""Data layer.

`generate_synthetic_matches` builds a coherent, chronological match history so
the whole pipeline runs with zero external data. Each player has latent skill
(overall + per-surface) that drives realistic serve/return rates; matches are
simulated with the very Markov model we predict with, so a well-built,
calibrated model should look good here. Real data is messier -- see
`load_sackmann_csv` at the bottom for how to swap it in.

Output columns:
    idx, surface, player_a, player_b, a_won,
    a_spw, a_rpw, b_spw, b_rpw       (serve/return point rates observed that match)
"""

import numpy as np
import pandas as pd

from markov import markov_match_prob

SURFACES = ["hard", "clay", "grass"]
TOUR_AVG_SPW = 0.64  # tour-average service points won


def generate_synthetic_matches(n_players: int = 60, n_matches: int = 6000,
                               seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    players = [f"P{i:03d}" for i in range(n_players)]

    skill = {p: rng.normal(0, 1) for p in players}  # overall latent skill
    surf_off = {(p, s): rng.normal(0, 0.4)          # per-surface adjustment
                for p in players for s in SURFACES}

    def rates(p, s):
        """Serve/return point rates from latent surface-adjusted skill."""
        z = skill[p] + surf_off[(p, s)]
        spw = np.clip(TOUR_AVG_SPW + 0.05 * z, 0.50, 0.80)
        rpw = np.clip((1 - TOUR_AVG_SPW) + 0.04 * z, 0.20, 0.50)
        return spw, rpw

    rows = []
    for idx in range(n_matches):
        s = rng.choice(SURFACES)
        a, b = rng.choice(players, size=2, replace=False)
        spw_a, rpw_a = rates(a, s)
        spw_b, rpw_b = rates(b, s)

        # "true" win prob from the Markov model on true rates -> draw a winner
        true_p = markov_match_prob(spw_a, rpw_a, spw_b, rpw_b, TOUR_AVG_SPW, best_of=3)
        a_won = rng.random() < true_p

        # observed match-level rates = true rates + measurement noise
        noise = lambda x: float(np.clip(x + rng.normal(0, 0.03), 0.05, 0.95))
        rows.append({
            "idx": idx, "surface": s, "player_a": a, "player_b": b,
            "a_won": bool(a_won), "true_p": float(true_p),
            "a_spw": noise(spw_a), "a_rpw": noise(rpw_a),
            "b_spw": noise(spw_b), "b_rpw": noise(rpw_b),
        })

    return pd.DataFrame(rows)


def load_sackmann_csv(path: str) -> pd.DataFrame:
    """Adapter for Jeff Sackmann's tennis_atp match CSVs.

    Get the data: https://github.com/JeffSackmann/tennis_atp (atp_matches_YYYY.csv).
    Those files are one row per match with winner_*/loser_* serve stats. We map
    them into this pipeline's a/b schema (A = winner here, so a_won is always
    True -- shuffle/relabel before modelling if you want balanced labels).

    Service points won  = (w_1stWon + w_2ndWon) / w_svpt
    Return points won    = 1 - opponent's service points won
    """
    df = pd.read_csv(path)

    def spw(prefix):
        return (df[f"{prefix}_1stWon"] + df[f"{prefix}_2ndWon"]) / df[f"{prefix}_svpt"]

    w_spw, l_spw = spw("w"), spw("l")
    out = pd.DataFrame({
        "idx": range(len(df)),
        "surface": df["surface"].str.lower(),
        "player_a": df["winner_name"],
        "player_b": df["loser_name"],
        "a_won": True,
        "a_spw": w_spw,
        "a_rpw": 1 - l_spw,   # winner's return rate = 1 - loser's serve rate
        "b_spw": l_spw,
        "b_rpw": 1 - w_spw,
    })
    return out.dropna().reset_index(drop=True)
