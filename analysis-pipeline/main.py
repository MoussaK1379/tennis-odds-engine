"""End-to-end tennis prediction pipeline.

Run:  python main.py

Walks a chronological match history, updating ELO online and tracking each
player's serve/return form. For every match (after a warmup) it produces three
probabilities -- ELO, Markov, and their ensemble -- then calibrates the
ensemble on a held-out slice and reports raw-vs-calibrated metrics on a final
test slice it never touched. Ends with a small, deliberately honest betting
demo wiring up de-vig -> EV -> fractional Kelly.
"""

import numpy as np

from data import generate_synthetic_matches, TOUR_AVG_SPW
from elo import EloModel
from markov import markov_match_prob
from betting import devig_two_way, expected_value, kelly_fraction, overround
from calibration import (
    brier_score, log_loss_score, expected_calibration_error,
    reliability_table, PlattCalibrator, save_reliability_plot,
)


class FormTracker:
    """Expanding mean of each player's observed serve/return rates."""

    def __init__(self, default_spw: float, default_rpw: float):
        self.d_spw, self.d_rpw = default_spw, default_rpw
        self.s: dict[str, list[float]] = {}  # player -> [sum_spw, sum_rpw, n]

    def get(self, p: str) -> tuple[float, float]:
        if p not in self.s or self.s[p][2] == 0:
            return self.d_spw, self.d_rpw
        ss, rs, n = self.s[p]
        return ss / n, rs / n

    def update(self, p: str, spw: float, rpw: float) -> None:
        if p not in self.s:
            self.s[p] = [0.0, 0.0, 0]
        self.s[p][0] += spw
        self.s[p][1] += rpw
        self.s[p][2] += 1


def run():
    df = generate_synthetic_matches()
    n = len(df)
    warmup_end = int(0.40 * n)   # ratings settle, no predictions kept
    calib_end = int(0.70 * n)    # held-out slice used only to fit calibrator
    print(f"matches: {n}  |  warmup<{warmup_end}  calib[{warmup_end}:{calib_end}]  "
          f"test[{calib_end}:]\n")

    elo = EloModel(surface_weight=0.5)
    form = FormTracker(TOUR_AVG_SPW, 1 - TOUR_AVG_SPW)

    calib = {"raw": [], "y": []}
    test = {"elo": [], "markov": [], "raw": [], "y": [], "true_p": []}

    for row in df.itertuples(index=False):
        a, b, s = row.player_a, row.player_b, row.surface

        # ---- predict using only pre-match state (no look-ahead) ----
        p_elo = elo.predict(a, b, s)
        a_spw, a_rpw = form.get(a)
        b_spw, b_rpw = form.get(b)
        p_mkv = markov_match_prob(a_spw, a_rpw, b_spw, b_rpw, TOUR_AVG_SPW, best_of=3)
        p_ens = 0.5 * (p_elo + p_mkv)
        y = 1 if row.a_won else 0

        if warmup_end <= row.idx < calib_end:
            calib["raw"].append(p_ens); calib["y"].append(y)
        elif row.idx >= calib_end:
            test["elo"].append(p_elo); test["markov"].append(p_mkv)
            test["raw"].append(p_ens); test["y"].append(y)
            test["true_p"].append(row.true_p)

        # ---- now reveal the result and update state ----
        elo.update(a, b, s, row.a_won)
        form.update(a, row.a_spw, row.a_rpw)
        form.update(b, row.b_spw, row.b_rpw)

    for k in calib:
        calib[k] = np.array(calib[k])
    for k in test:
        test[k] = np.array(test[k])

    # ---- calibrate on the held-out slice, apply to test ----
    platt = PlattCalibrator().fit(calib["raw"], calib["y"])
    p_cal = platt.transform(test["raw"])

    # ---- metrics ----
    def report(name, p, y):
        print(f"{name:<22} logloss={log_loss_score(p, y):.4f}  "
              f"brier={brier_score(p, y):.4f}  "
              f"ECE={expected_calibration_error(p, y):.4f}")

    y = test["y"]
    print("=== test-set metrics (lower is better) ===")
    report("ELO only", test["elo"], y)
    report("Markov only", test["markov"], y)
    report("Ensemble (raw)", test["raw"], y)
    report("Ensemble (calibrated)", p_cal, y)

    print("\n=== reliability: ensemble raw vs calibrated ===")
    print(f"{'bin':<12}{'raw pred':>9}{'raw obs':>9}{'cal pred':>10}{'cal obs':>9}{'n':>7}")
    raw_tbl = {r[0]: r for r in reliability_table(test["raw"], y)}
    cal_tbl = {r[0]: r for r in reliability_table(p_cal, y)}
    for b in sorted(set(raw_tbl) | set(cal_tbl)):
        r = raw_tbl.get(b); c = cal_tbl.get(b)
        print(f"{b:<12}"
              f"{(r[1] if r else float('nan')):>9.3f}{(r[2] if r else float('nan')):>9.3f}"
              f"{(c[1] if c else float('nan')):>10.3f}{(c[2] if c else float('nan')):>9.3f}"
              f"{(r[3] if r else (c[3] if c else 0)):>7}")

    save_reliability_plot(test["raw"], y, "reliability_raw.png", title="Ensemble (raw)")
    save_reliability_plot(p_cal, y, "reliability_calibrated.png", title="Ensemble (calibrated)")

    betting_demo(p_cal, y, test["true_p"])


def betting_demo(p_cal, y, true_p, vig=0.045, market_noise=0.05, seed=11):
    """Honest wiring demo. The simulated market is an INDEPENDENT, noisy
    estimate of the true probability (not a copy of our own number), plus vig.
    Edge then exists only where our model lands closer to the truth than the
    market's price does -- exactly the real-world condition. On synthetic data
    our model can be genuinely sharp, but a real market is sharper and the edge
    is thinner; treat the ROI as mechanics, not a forecast."""
    rng = np.random.default_rng(seed)
    bankroll, start = 1000.0, 1000.0
    flat_profit, flat_staked, n_bets = 0.0, 0.0, 0

    print("\n=== betting demo (simulated market, illustrative only) ===")
    for p, won, tp in zip(p_cal, y, true_p):
        # market = independent noisy estimate of the TRUE prob, then add margin
        fair = float(np.clip(tp + rng.normal(0, market_noise), 0.02, 0.98))
        q_a, q_b = fair + vig / 2, (1 - fair) + vig / 2
        odds_a, odds_b = 1 / q_a, 1 / q_b

        # evaluate both sides with the calibrated probability
        for side_p, odds, win in ((p, odds_a, won == 1), (1 - p, odds_b, won == 0)):
            if expected_value(side_p, odds) > 0:
                n_bets += 1
                flat_profit += (odds - 1) if win else -1
                flat_staked += 1
                stake = kelly_fraction(side_p, odds, fraction=0.25) * bankroll
                bankroll += stake * (odds - 1) if win else -stake

    print(f"avg market overround: {vig*100:.1f}%   bets placed: {n_bets}")
    if flat_staked:
        print(f"flat 1u ROI:    {100*flat_profit/flat_staked:+.2f}%  "
              f"(profit {flat_profit:+.1f}u over {flat_staked:.0f}u staked)")
    print(f"1/4-Kelly bankroll: {start:.0f} -> {bankroll:.0f}  "
          f"({100*(bankroll-start)/start:+.2f}%)")


if __name__ == "__main__":
    run()
