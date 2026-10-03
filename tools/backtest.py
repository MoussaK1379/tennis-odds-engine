"""Walk-forward backtest and tuning for the Tennis Odds Engine model.

Every cached match (data/api_tennis_matches.csv) is predicted using only the
matches before it, then fed to the model. Scores are computed on finished
matches (retirements and walkovers are not scored):

    log loss  - main score; lower is better. Coin-flip guessing scores 0.693.
    Brier     - mean squared error of the probability; lower is better.
    accuracy  - share of matches where the favourite won.

Periods: the first six months only warm the ratings up; settings are tuned on
the next twelve months; the most recent months are a holdout that tuning
never sees, so the holdout numbers are the honest ones.

    python tools/backtest.py                 # compare original vs current model_params.json
    python tools/backtest.py --tune          # grid-search, write model_params.json + report
"""

import argparse, itertools, json, math, os, sys, time
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model                                     # noqa: E402
import update_data                               # noqa: E402

WARM_END = date(2025, 3, 31)
TUNE = (date(2025, 4, 1), date(2026, 3, 31))
HOLD_START = date(2026, 4, 1)

MIN_GAIN = 0.001   # log loss a more complex option must save before it is adopted

BASELINE = model.Params(rating="legacy", k0=250, k_off=5, k_exp=0.4, half_life=365, shrink=400,
                        surf_shrink=0, court_shrink=0, w_atp=0.5, w_wta=0.5, n0=0)


def load(cache_path):
    cache = update_data.load_api_cache(cache_path)
    out = {}
    for tour in ("ATP", "WTA"):
        ms = update_data.from_api_cache(cache, tour)
        ms.sort(key=lambda m: (m["date"], m["order"]))
        out[tour] = ms
    return out


def run(P, data, set_prob):
    """-> list of (tour, date, y, p, p_elo, p_mkv, n, prior_min, qualifying)."""
    recs = []
    for tour, ms in data.items():
        eng = model.TourEngine(tour, P, set_prob=set_prob)
        seen = {}
        for m in ms:
            if m["status"] == "completed":
                a, b = sorted((m["winner"], m["loser"]))
                r = eng.predict(a, b, m["surface"], m["best_of"], m["date"], m.get("court"))
                recs.append((tour, m["date"], 1 if a == m["winner"] else 0, r["p"], r["p_elo"],
                             r["p_mkv"], r["n"], min(seen.get(a, 0), seen.get(b, 0)), m["qualifying"]))
            eng.update(m)
            seen[m["winner"]] = seen.get(m["winner"], 0) + 1
            seen[m["loser"]] = seen.get(m["loser"], 0) + 1
    return recs


def score(recs, lo, hi, col=3, tour=None, min_prior=0):
    n = ll = br = acc = 0
    for r in recs:
        if not (lo <= r[1] <= hi) or (tour and r[0] != tour) or r[7] < min_prior:
            continue
        p = min(max(r[col], 1e-6), 1 - 1e-6)
        y = r[2]
        n += 1
        ll -= math.log(p if y else 1 - p)
        br += (p - y) ** 2
        acc += (p > 0.5) == (y == 1) if p != 0.5 else 0.5
    return {"n": n, "logloss": ll / n, "brier": br / n, "acc": acc / n} if n else {"n": 0}


def blended(recs, w_atp, w_wta, n0):
    out = []
    for r in recs:
        w = w_atp if r[0] == "ATP" else w_wta
        if n0 > 0:
            w *= r[6] / (r[6] + n0)
        out.append(r[:3] + (w * r[5] + (1 - w) * r[4],) + r[4:])
    return out


def calibration(recs, lo, hi, bins=10):
    """Predicted vs actual win rate, folded so the favourite is always 'A'."""
    b = [[0, 0.0, 0] for _ in range(bins // 2)]
    for r in recs:
        if not (lo <= r[1] <= hi):
            continue
        p, y = r[3], r[2]
        if p < 0.5:
            p, y = 1 - p, 1 - y
        i = min(int((p - 0.5) / (0.5 / len(b))), len(b) - 1)
        b[i][0] += 1; b[i][1] += p; b[i][2] += y
    return [{"range": f"{50 + i * 100 // bins}-{50 + (i + 1) * 100 // bins}%", "n": n,
             "predicted": round(sp / n, 3), "actual": round(sy / n, 3)}
            for i, (n, sp, sy) in enumerate(b) if n]


def fmt(s):
    return (f"log loss {s['logloss']:.4f} · Brier {s['brier']:.4f} · favourite won {s['acc']:.1%} · n={s['n']}"
            if s.get("n") else "no matches")


def summary(name, recs, end):
    out = {"name": name}
    for label, lo, hi in (("tune", *TUNE), ("holdout", HOLD_START, end)):
        out[label] = {"all": score(recs, lo, hi), "ATP": score(recs, lo, hi, tour="ATP"),
                      "WTA": score(recs, lo, hi, tour="WTA"),
                      "established": score(recs, lo, hi, min_prior=10),
                      "elo_only": score(recs, lo, hi, col=4), "serve_only": score(recs, lo, hi, col=5)}
    out["calibration_holdout"] = calibration(recs, HOLD_START, end)
    return out


def tune(data, set_prob, log):
    t0 = time.time()
    lo, hi = TUNE
    best = {}

    # Stage 1: rating system on the rating-only prediction.
    cands = [BASELINE]
    for k0, ke, sk in itertools.product((100, 150, 200, 250), (0.3, 0.4, 0.5), (0.1, 0.25, 0.5)):
        cands.append(model.Params(rating="elo", k0=k0, k_exp=ke, surf_k=sk))
    for tau, gd, rd0, gsk in itertools.product((0.2, 0.3, 0.5), (10, 15, 30), (150, 200, 250, 350), (0, 5, 10, 20)):
        cands.append(model.Params(rating="glicko", g_tau=tau, g_days=gd, g_rd0=rd0, g_surf_k=gsk))
    res = []
    for P in cands:
        recs = run(P, data, set_prob)
        res.append((score(recs, lo, hi, col=4)["logloss"], P))
    res.sort(key=lambda x: x[0])
    top = {}
    for kind in ("legacy", "elo", "glicko"):
        top[kind] = next(x for x in res if x[1].rating == kind)
        log(f"  best {kind:6}: rating-only log loss {top[kind][0]:.4f}  {rating_desc(top[kind][1])}")
    # Glicko-2 is the more complex system: adopt it only for a real gain.
    best_rating = (top["glicko"] if top["glicko"][0] <= top["elo"][0] - MIN_GAIN else top["elo"])[1]
    log(f"Stage 1 (rating) done in {time.time() - t0:.0f}s -> {best_rating.rating}")

    # Stage 2: serve model on the serve-only prediction (independent of the rating).
    res = []
    for hl, sh, ss, cs in itertools.product((90, 135, 180, 365), (200, 400, 800),
                                            (0, 2000, 4000, 8000, 16000), (0, 3000)):
        P = model.Params(rating="legacy", half_life=hl, shrink=sh, surf_shrink=ss, court_shrink=cs)
        recs = run(P, data, set_prob)
        res.append((score(recs, lo, hi, col=5)["logloss"], P))
    res.sort(key=lambda x: x[0])
    with_court = next(x for x in res if x[1].court_shrink > 0)
    no_court = next(x for x in res if x[1].court_shrink == 0)
    for label, (ll, b) in (("with court speed", with_court), ("without", no_court)):
        log(f"  best serve model {label}: serve-only log loss {ll:.4f}  half-life {b.half_life:.0f}d, "
            f"shrink {b.shrink:.0f}, surface shrink {b.surf_shrink:.0f}, court shrink {b.court_shrink:.0f}")
    # The court adjustment adds a table to maintain: adopt it only for a real gain.
    b = (with_court if with_court[0] <= no_court[0] - MIN_GAIN else no_court)[1]
    log(f"Stage 2 (serve) done in {time.time() - t0:.0f}s")

    # Stage 3: blend weights on the combined prediction.
    P = model.Params(**{**best_rating.to_dict(),
                        **{k: getattr(b, k) for k in ("half_life", "shrink", "surf_shrink", "court_shrink")}})
    recs = run(P, data, set_prob)
    res = []
    grid = [i / 20 for i in range(21)]
    for n0 in (0, 250, 500, 1000, 2000, 4000):
        for wa, ww in itertools.product(grid, grid):
            res.append((score(blended(recs, wa, ww, n0), lo, hi)["logloss"], wa, ww, n0))
    res.sort()
    _, wa, ww, n0 = res[0]
    P.w_atp, P.w_wta, P.n0 = wa, ww, n0
    log(f"  best blend: serve-model weight ATP {wa:.2f}, WTA {ww:.2f}, thin-sample half-point {n0} serve points")
    log(f"Stage 3 (blend) done in {time.time() - t0:.0f}s")
    return P


def rating_desc(P):
    if P.rating == "glicko":
        return f"tau {P.g_tau}, {P.g_days:.0f}-day periods, start RD {P.g_rd0:.0f}, surface step {P.g_surf_k:.0f}"
    if P.rating == "elo":
        return f"K {P.k0:.0f}/(n+{P.k_off:.0f})^{P.k_exp}, surface offset rate {P.surf_k}"
    return "original: separate surface Elos from 1500"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/api_tennis_matches.csv")
    ap.add_argument("--params", default="model_params.json")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--report", default="backtest_report.json")
    args = ap.parse_args()

    data = load(args.cache)
    end = max(m["date"] for ms in data.values() for m in ms)
    log = lambda s: print(s, file=sys.stderr)
    log(f"{sum(len(v) for v in data.values())} matches, {min(m['date'] for ms in data.values() for m in ms)} -> {end}")
    set_prob = model.SetTable()

    if args.tune:
        P = tune(data, set_prob, log)
        with open(args.params, "w") as fh:
            json.dump({"version": str(date.today()), "tuned_on": f"{TUNE[0]}..{TUNE[1]}",
                       "params": P.to_dict()}, fh, indent=2)
        log(f"wrote {args.params}")
    else:
        with open(args.params) as fh:
            P = model.Params.from_dict(json.load(fh)["params"])

    base = summary("original model", run(BASELINE, data, set_prob), end)
    new = summary("tuned model", run(P, data, set_prob), end)
    for s in (base, new):
        log(f"\n{s['name']}")
        for label in ("tune", "holdout"):
            log(f"  {label:8} all    {fmt(s[label]['all'])}")
            for k in ("ATP", "WTA", "established", "elo_only", "serve_only"):
                log(f"  {'':8} {k:6} {fmt(s[label][k])}")
    log("\nholdout calibration (tuned): " + ", ".join(
        f"{c['range']}: predicted {c['predicted']:.0%} / won {c['actual']:.0%} (n={c['n']})" for c in new["calibration_holdout"]))
    with open(args.report, "w") as fh:
        json.dump({"periods": {"warm_up_until": str(WARM_END), "tune": [str(TUNE[0]), str(TUNE[1])],
                               "holdout": [str(HOLD_START), str(end)]},
                   "params": P.to_dict(), "original": base, "tuned": new}, fh, indent=1, default=str)
    log(f"wrote {args.report}")


if __name__ == "__main__":
    main()
