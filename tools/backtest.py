"""Walk-forward backtest and tuning for the Tennis Odds Engine model.

Every cached match (data/api_tennis_matches.csv) is predicted using only the
matches before it, then fed to the model. Scores are computed on finished
matches (retirements and walkovers are not scored):

    log loss  - main score; lower is better. Coin-flip guessing scores 0.693.
    Brier     - mean squared error of the probability; lower is better.
    accuracy  - share of matches where the favourite won.

Periods: the first twelve months only warm the ratings up; settings are tuned on
the next six months; the most recent months are a holdout that tuning
never sees, so the holdout numbers are the honest ones.

    python tools/backtest.py                 # compare original vs current model_params.json
    python tools/backtest.py --tune          # tune all settings jointly, write model_params.json + report
    python tools/backtest.py --challenger    # does Challenger history help? (--write to adopt)

Scores are on tour-level matches unless they say Challenger, so a model fed Challenger
history is judged on exactly the same matches as one without it.
"""

import argparse, itertools, json, math, os, sys, time
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model                                     # noqa: E402
import update_data                               # noqa: E402

# The cached history starts 2024-09-30 and the ratings take about a year to settle (log
# loss falls from 0.68 to 0.61 over the first twelve months), so tuning starts after that:
# tuned on the settling-in months, it favours settings that forget fast and suit
# newcomers, which then lose on mature ratings.
WARM_END = date(2025, 9, 30)
TUNE = (date(2025, 10, 1), date(2026, 3, 31))
HOLD_START = date(2026, 4, 1)

MIN_GAIN = 0.001   # log loss a more complex option must save before it is adopted

BASELINE = model.Params(rating="legacy", k0=250, k_off=5, k_exp=0.4, half_life=365, shrink=400,
                        surf_shrink=0, court_shrink=0, w_atp=0.5, w_wta=0.5, n0=0)


def load(cache_path):
    """Both tours, Challenger level included (marked level 'challenger'); run() leaves
    it out unless the settings use it."""
    cache = update_data.load_api_cache(cache_path)
    out = {}
    for tour in ("ATP", "WTA"):
        ms = update_data.from_api_cache(cache, tour, challenger=True)
        ms.sort(key=lambda m: (m["date"], m["order"]))
        out[tour] = ms
    return out


def run(P, data, set_prob):
    """-> list of (tour, date, y, p, p_elo, p_mkv, n, prior_min, qualifying, stack features,
    (sets won by a, sets won by b) or None, best of)."""
    recs = []
    for tour, ms in data.items():
        eng = model.TourEngine(tour, P, set_prob=set_prob)
        seen = {}
        for m in ms:
            if m.get("level") == "challenger" and not P.ch_use:
                continue
            if m["status"] == "completed":
                a, b = sorted((m["winner"], m["loser"]))
                r = eng.predict(a, b, m["surface"], m["best_of"], m["date"], m.get("court"), m["qualifying"])
                recs.append((tour, m["date"], 1 if a == m["winner"] else 0, r["p"], r["p_elo"],
                             r["p_mkv"], r["n"], min(seen.get(a, 0), seen.get(b, 0)), m["qualifying"],
                             r["feats"], _sets_for(m, a), m["best_of"], r["pa"], r["pb"], _set1_for(m, a),
                             m.get("level", "tour")))
            eng.update(m)
            seen[m["winner"]] = seen.get(m["winner"], 0) + 1
            seen[m["loser"]] = seen.get(m["loser"], 0) + 1
    return recs


def _sets_for(m, a):
    if not m.get("sets"):
        return None
    w, l = m["sets"]
    return (w, l) if a == m["winner"] else (l, w)


def _set1_for(m, a):
    if not m.get("set1"):
        return None
    w, l = m["set1"]
    return (w, l) if a == m["winner"] else (l, w)


def first_set_check(recs, P, end, log):
    """How well the first-set exact game scores match reality on the holdout."""
    rows = [r for r in recs if HOLD_START <= r[1] <= end and r[14] and r[10] and r[15] == "tour"]
    if len(rows) < 200:
        log(f"  first-set game scores: only {len(rows)} matches with game scores, skipped")
        return None
    ll = tb_pred = tb_act = 0.0
    for r in rows:
        m = {"ATP": {3: P.set_m_atp3, 5: P.set_m_atp5}, "WTA": {3: P.set_m_wta3}}[r[0]].get(r[11], 0.0)
        dist = model.set_dist(r[3], r[11], m)
        f = model.first_set_scores(r[12], r[13], model.first_set_chance(dist, r[11], m))
        ll -= math.log(max(f.get(tuple(r[14]), 1e-9), 1e-9))
        tb_pred += f.get((7, 6), 0) + f.get((6, 7), 0)
        tb_act += r[14] in ((7, 6), (6, 7))
    out = {"n": len(rows), "logloss": ll / len(rows), "tiebreak_predicted": tb_pred / len(rows),
           "tiebreak_actual": tb_act / len(rows)}
    log(f"  first-set game scores on holdout: log loss {out['logloss']:.3f} (14 outcomes; guessing evenly = "
        f"{math.log(14):.3f}); tiebreak sets predicted {out['tiebreak_predicted']:.1%}, actual "
        f"{out['tiebreak_actual']:.1%} (n={len(rows)})")
    return out


SET_GRID = [round(-0.2 + 0.05 * i, 2) for i in range(29)]     # -0.2 .. 1.2


def set_rows(recs, lo, hi, tour, bo):
    need = 3 if bo == 5 else 2
    return [(r[3], r[10]) for r in recs if r[0] == tour and r[11] == bo and lo <= r[1] <= hi
            and r[15] == "tour" and r[10] and max(r[10]) == need and min(r[10]) < need]


def set_logloss(rows, bo, m):
    ll = 0.0
    for p, sc in rows:
        ll -= math.log(max(model.set_dist(p, bo, m).get(tuple(sc), 1e-9), 1e-9))
    return ll / len(rows) if rows else float("nan")


def straight_sets(rows, bo, m):
    """(predicted, actual) share of matches won without dropping a set."""
    need = 3 if bo == 5 else 2
    pred = sum(model.set_dist(p, bo, m).get((need, 0), 0) + model.set_dist(p, bo, m).get((0, need), 0)
               for p, _ in rows)
    act = sum(1 for _, sc in rows if min(sc) == 0)
    return pred / len(rows), act / len(rows)


def fit_sets(recs, end, log, P):
    """Stage 5: set momentum per tour and format, fitted on the tuning year."""
    lo, hi = TUNE
    out = {}
    for tour, bo, field in (("ATP", 3, "set_m_atp3"), ("ATP", 5, "set_m_atp5"), ("WTA", 3, "set_m_wta3")):
        tune_rows, hold_rows = set_rows(recs, lo, hi, tour, bo), set_rows(recs, HOLD_START, end, tour, bo)
        if len(tune_rows) < 200:
            log(f"  sets {tour} bo{bo}: only {len(tune_rows)} matches with set scores, keeping momentum 0")
            continue
        best = min(SET_GRID, key=lambda m: set_logloss(tune_rows, bo, m))
        t0_, t1_ = set_logloss(tune_rows, bo, 0.0), set_logloss(tune_rows, bo, best)
        h0, h1 = set_logloss(hold_rows, bo, 0.0), set_logloss(hold_rows, bo, best)
        sp0, sa = straight_sets(hold_rows, bo, 0.0)
        sp1, _ = straight_sets(hold_rows, bo, best)
        adopt = t1_ <= t0_ - MIN_GAIN
        log(f"  sets {tour} bo{bo}: momentum {best:+.2f}; exact-score log loss tune {t0_:.4f} -> {t1_:.4f}, "
            f"holdout {h0:.4f} -> {h1:.4f}; straight sets on holdout: predicted {sp0:.1%} (independent) / "
            f"{sp1:.1%} (with momentum), actual {sa:.1%} (n={len(hold_rows)})" + ("" if adopt else " -> not adopted"))
        out[field] = best if adopt else 0.0
        setattr(P, field, out[field])
        out[f"{tour}{bo}"] = {"momentum": best, "tune": [t0_, t1_], "holdout": [h0, h1],
                              "straight_sets_holdout": {"independent": sp0, "momentum": sp1, "actual": sa},
                              "n_holdout": len(hold_rows), "adopted": adopt}
    return out


def score(recs, lo, hi, col=3, tour=None, min_prior=0, level="tour"):
    """level: 'tour' (default), 'challenger', or None for both."""
    n = ll = br = acc = 0
    for r in recs:
        if not (lo <= r[1] <= hi) or (tour and r[0] != tour) or r[7] < min_prior:
            continue
        if level and r[15] != level:
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
        if not (lo <= r[1] <= hi) or r[15] != "tour":
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
                      "elo_only": score(recs, lo, hi, col=4), "serve_only": score(recs, lo, hi, col=5),
                      "challenger": score(recs, lo, hi, level="challenger")}
    out["calibration_holdout"] = calibration(recs, HOLD_START, end)
    return out


# Settings searched by tune(), one at a time (coordinate descent), each judged on the
# finished prediction: the stack refitted on the tuning year for every candidate.
TUNE_GRID = {
    "shrink": [25, 50, 100, 200, 400],
    "surf_shrink": [250, 500, 1000, 2000, 4000],
    "half_life": [60, 90, 135, 200, 300],
    "k0": [100, 150, 200, 250, 300],
    "k_exp": [0.3, 0.4, 0.5, 0.6],
    "k_off": [1, 2, 5, 10],
    "surf_k": [0.1, 0.25, 0.5, 0.75, 1.0],
    "init_elo": [1350, 1400, 1450, 1500],
    "init_elo_qual": [1100, 1150, 1200, 1250, 1300],
    "init_elo_ch": [1100, 1150, 1200, 1250],
    "ch_k": [0.5, 0.75, 1.0],
    "ch_sv": [0.5, 0.75, 1.0],
    "court_shrink": [0, 3000, 10000],
}


def fit_stacks(recs, lo, hi):
    """Stack weights per tour, fitted on tour-level matches between lo and hi."""
    return {tour: fit_logistic([(r[9], r[2]) for r in recs
                                if r[0] == tour and lo <= r[1] <= hi and r[15] == "tour"])
            for tour in ("ATP", "WTA")}


def stacked(recs, stacks):
    return [r[:3] + (model.stack_prob(stacks[r[0]], r[9]),) + r[4:] for r in recs]


def best_blend(recs, lo, hi):
    """Best serve-model weights per tour for the plain blend: (log loss, w_atp, w_wta)."""
    grid = [i / 20 for i in range(21)]
    return min((score(blended(recs, wa, ww, 0), lo, hi)["logloss"], wa, ww)
               for wa, ww in itertools.product(grid, grid))


def tune(data, set_prob, log, start=None, passes=3):
    """Coordinate descent over TUNE_GRID from `start` (the saved settings): each setting in
    turn takes its best grid value, judged on the tuning year's log loss with the stack
    refitted, until a full pass changes nothing. Tuning every setting on the finished
    prediction finds combinations a stage-by-stage search misses (e.g. light serve
    shrinkage pays off once the stack knows how thin each sample is)."""
    t0 = time.time()
    lo, hi = TUNE
    P0 = model.Params.from_dict({**(start or model.Params()).to_dict(), "rating": "elo",
                                 "stack_atp": None, "stack_wta": None})
    cache = {}

    def evaluate(P):
        k = json.dumps(P.to_dict(), sort_keys=True, default=str)
        if k not in cache:
            recs = run(P, data, set_prob)
            cache[k] = score(stacked(recs, fit_stacks(recs, lo, hi)), lo, hi)["logloss"]
        return cache[k]

    best, P = evaluate(P0), P0
    log(f"  start: stacked log loss {best:.4f} on the tuning year")
    for i in range(passes):
        moved = False
        for name, grid in TUNE_GRID.items():
            for v in grid:
                cand = model.Params.from_dict({**P.to_dict(), name: v})
                ll = evaluate(cand)
                if ll < best - 1e-5:
                    best, P, moved = ll, cand, True
            log(f"  pass {i + 1}, {name:13} -> {getattr(P, name)}  (log loss {best:.4f}, {time.time() - t0:.0f}s)")
        if not moved:
            break

    recs = run(P, data, set_prob)
    stacks = fit_stacks(recs, lo, hi)
    stack_ll = score(stacked(recs, stacks), lo, hi)["logloss"]
    blend_ll, wa, ww = best_blend(recs, lo, hi)
    P.w_atp, P.w_wta, P.n0 = wa, ww, 0
    for tour in ("ATP", "WTA"):
        log(f"  stack {tour}: " + ", ".join(f"{n} {w:+.3f}" for n, w in zip(model.STACK_FEATURES, stacks[tour])))
    log(f"  stack log loss {stack_ll:.4f} vs blend {blend_ll:.4f} (serve weight ATP {wa:.2f}, WTA {ww:.2f}) "
        "on the tuning year")
    # The stack is the more complex option: adopt it only for a real gain.
    if stack_ll <= blend_ll - MIN_GAIN:
        P.stack_atp = [round(w, 5) for w in stacks["ATP"]]
        P.stack_wta = [round(w, 5) for w in stacks["WTA"]]
        log("  -> stack adopted")
    else:
        log(f"  -> stack not adopted (not {MIN_GAIN} better)")
    log(f"tuning done in {time.time() - t0:.0f}s")
    return P


def fit_logistic(rows, l2=1.0, iters=25):
    """No-intercept logistic regression by Newton's method with a small L2 penalty.
    rows: [(features, y)]. Pure Python so the backtest has no dependencies."""
    k = len(rows[0][0])
    w = [0.0] * k
    for _ in range(iters):
        g = [l2 * wi for wi in w]
        H = [[(l2 if i == j else 0.0) for j in range(k)] for i in range(k)]
        for x, y in rows:
            p = model.stack_prob(w, x)
            e, v = p - y, p * (1 - p)
            for i in range(k):
                g[i] += e * x[i]
                for j in range(i, k):
                    H[i][j] += v * x[i] * x[j]
        for i in range(k):
            for j in range(i):
                H[i][j] = H[j][i]
        step = _solve(H, g)
        w = [wi - si for wi, si in zip(w, step)]
        if max(abs(si) for si in step) < 1e-7:
            break
    return w


def _solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        for r in range(n):
            if r != c and M[r][c]:
                f = M[r][c] / M[c][c]
                M[r] = [a - f * bb for a, bb in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def challenger_study(data, set_prob, base, end, log):
    """Feed Challenger results into the rating pool and pick the settings that help the
    tour-level tune score most. -> (best Params, report). Adopted only if the gain on the
    tune period passes MIN_GAIN; the holdout is reported, never used to choose."""
    lo, hi = TUNE
    recs0 = run(base, data, set_prob)
    s0 = score(recs0, lo, hi)
    log(f"without Challengers: tune {fmt(s0)}")
    log(f"                     holdout {fmt(score(recs0, HOLD_START, end))}")
    n_ch = sum(1 for ms in data.values() for m in ms if m.get("level") == "challenger")
    if not n_ch:
        log("no Challenger matches cached yet")
        return base, {"challenger_matches": 0}
    trials = []
    for ch_k, ch_sv, init in itertools.product((0.5, 0.75, 1.0), (0.5, 1.0), (1150, 1250, 1350)):
        P = model.Params.from_dict({**base.to_dict(), "ch_use": True, "ch_k": ch_k, "ch_sv": ch_sv,
                                    "init_elo_ch": init})
        recs = run(P, data, set_prob)
        st = score(recs, lo, hi)
        trials.append((st["logloss"], P, recs))
        log(f"  ch_k {ch_k:<4} ch_sv {ch_sv:<3} start {init}: tune {fmt(st)}")
    trials.sort(key=lambda t: t[0])
    ll, best, recs = trials[0]
    adopt = s0["logloss"] - ll >= MIN_GAIN
    rep = {"challenger_matches": n_ch,
           "without": {"tune": s0, "holdout": score(recs0, HOLD_START, end),
                       "holdout_established": score(recs0, HOLD_START, end, min_prior=10)},
           "with": {"settings": {k: getattr(best, k) for k in ("ch_k", "ch_sv", "init_elo_ch")},
                    "tune": score(recs, lo, hi), "holdout": score(recs, HOLD_START, end),
                    "holdout_established": score(recs, HOLD_START, end, min_prior=10),
                    "challenger_tune": score(recs, lo, hi, level="challenger"),
                    "challenger_holdout": score(recs, HOLD_START, end, level="challenger")},
           "adopted": adopt}
    log(f"best with Challengers ({rep['with']['settings']}): tune {fmt(rep['with']['tune'])}")
    log(f"                     holdout {fmt(rep['with']['holdout'])}")
    log(f"Challenger matches themselves: holdout {fmt(rep['with']['challenger_holdout'])}")
    log("-> " + ("use Challenger history" if adopt else "not enough gain; keep tour-level only"))
    return (best if adopt else base), rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/api_tennis_matches.csv")
    ap.add_argument("--params", default="model_params.json")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--sets", action="store_true", help="only (re)fit set momentum, keeping the other settings")
    ap.add_argument("--report", default="backtest_report.json")
    ap.add_argument("--challenger", action="store_true", help="test feeding Challenger results into the ratings")
    ap.add_argument("--write", action="store_true", help="with --challenger: save the result to --params")
    args = ap.parse_args()

    data = load(args.cache)
    end = max(m["date"] for ms in data.values() for m in ms)
    log = lambda s: print(s, file=sys.stderr)
    log(f"{sum(len(v) for v in data.values())} matches, {min(m['date'] for ms in data.values() for m in ms)} -> {end}")
    set_prob = model.SetTable()

    try:
        with open(args.params) as fh:
            saved = json.load(fh)
    except (OSError, ValueError):
        saved = {}
    if args.challenger:
        P0 = model.Params.from_dict(saved["params"])
        P, rep = challenger_study(data, set_prob, P0, end, log)
        if args.write:
            saved.update({"version": str(date.today()), "params": P.to_dict(), "challenger_study": rep})
            with open(args.params, "w") as fh:
                json.dump(saved, fh, indent=2, default=str)
            log(f"wrote {args.params}")
        return
    sets_report = None
    if args.tune or args.sets:
        if args.tune:
            P = tune(data, set_prob, log, start=model.Params.from_dict(saved.get("params")))
        else:
            P = model.Params.from_dict(saved["params"])
        recs_sets = run(P, data, set_prob)
        sets_report = fit_sets(recs_sets, end, log, P)
        sets_report["first_set"] = first_set_check(recs_sets, P, end, log)
        saved.update({"version": str(date.today()), "params": P.to_dict()})
        if args.tune:
            saved["tuned_on"] = f"{TUNE[0]}..{TUNE[1]}"
        with open(args.params, "w") as fh:
            json.dump(saved, fh, indent=2)
        log(f"wrote {args.params}")
    else:
        P = model.Params.from_dict(saved["params"])

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
    if sets_report is None and os.path.exists(args.report):
        try:                                     # keep the last set-momentum fit
            with open(args.report) as fh:
                sets_report = json.load(fh).get("sets")
        except ValueError:
            pass
    with open(args.report, "w") as fh:
        json.dump({"periods": {"warm_up_until": str(WARM_END), "tune": [str(TUNE[0]), str(TUNE[1])],
                               "holdout": [str(HOLD_START), str(end)]},
                   "params": P.to_dict(), "original": base, "tuned": new, "sets": sets_report},
                  fh, indent=1, default=str)
    log(f"wrote {args.report}")


if __name__ == "__main__":
    main()
