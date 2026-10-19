"""Model vs bookmakers on the past: does the model know anything the market doesn't?

Joins the walk-forward predictions of tools/backtest.py (same model, same settings,
every match predicted from earlier matches only) with the pre-match odds in
data/odds_history.csv (tools/fetch_odds.py), by api-tennis event key, and reports:

  1. sharpness: log loss of the model, the average de-vigged market and Pinnacle
  2. information: a logistic blend of market and model log-odds fitted on the tuning
     period; if the model's weight is ~0, the market already knows what it knows
  3. betting: flat 1-unit bets on the best available price whenever a probability
     (model, or the blend) says the price is value by more than a threshold, with
     ROI and a 90% bootstrap interval, overall and by tour, level and price range

The tuning period chooses the blend; the holdout is the honest test. With --write,
the blend is refitted on every match with odds (more data, more recent) and saved to
model_params.json as "market_blend": the probability update_matches.py uses for value
bets (model.bet_prob).

    python tools/market_backtest.py [--json market_report.json] [--write]
"""

import argparse, csv, json, math, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model                                     # noqa: E402
import update_data                               # noqa: E402
import backtest as bt                            # noqa: E402

THRESHOLDS = (0.0, 0.03, 0.06, 0.10)


def predictions(P, data, set_prob):
    """{event_key: (tour, date, level, player a, p(a wins) from the model)} for finished matches."""
    out = {}
    for tour, ms in data.items():
        eng = model.TourEngine(tour, P, set_prob=set_prob)
        for m in ms:
            if m.get("level") == "challenger" and not P.ch_use:
                continue
            if m["status"] == "completed":
                a, b = sorted((m["winner"], m["loser"]))
                r = eng.predict(a, b, m["surface"], m["best_of"], m["date"], m.get("court"), m["qualifying"])
                out[str(m["order"][1])] = (tour, m["date"], m.get("level", "tour"), a, m["winner"], r["p"])
            eng.update(m)
    return out


def joined(preds, odds_path, cache):
    """Rows: dict(tour, date, level, y, model, market, pin, price_a, price_b, books) from a's side."""
    rows = []
    with open(odds_path, encoding="utf-8") as fh:
        for o in csv.DictReader(fh):
            p = preds.get(o["event_key"])
            c = cache.get(o["event_key"])
            if not p or not c:
                continue
            tour, d, level, a, winner, pm = p
            first_won = str(o["first_key"]) == str(c["winner_key"])
            a_is_first = (a == winner) == first_won
            f1, pin = float(o["fair_1"]), (float(o["pin_1"]) if o["pin_1"] else None)
            best_1, best_2 = float(o["best_1"]), float(o["best_2"])
            rows.append({"tour": tour, "date": d, "level": level, "y": 1 if a == winner else 0, "model": pm,
                         "market": f1 if a_is_first else 1 - f1,
                         "pin": None if pin is None else (pin if a_is_first else 1 - pin),
                         "price_a": best_1 if a_is_first else best_2, "price_b": best_2 if a_is_first else best_1,
                         "books": int(o["books"])})
    return rows


def L(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def logloss(rows, key):
    rs = [r for r in rows if r[key] is not None]
    if not rs:
        return None, 0
    s = sum(-math.log(min(max(r[key] if r["y"] else 1 - r[key], 1e-6), 1)) for r in rs)
    return s / len(rs), len(rs)


def bets(rows, key, thr):
    """(price, won) for each value bet: probability x best price - 1 > thr, either side."""
    out = []
    for r in rows:
        p = r[key]
        for q, price, won in ((p, r["price_a"], r["y"] == 1), (1 - p, r["price_b"], r["y"] == 0)):
            if q * price - 1 > thr:
                out.append((price, won))
    return out


def roi(b):
    return sum((pr - 1) if w else -1 for pr, w in b) / len(b) if b else float("nan")


def ci(b, n=2000, seed=7):
    if len(b) < 20:
        return float("nan"), float("nan")
    rnd = random.Random(seed)
    rs = sorted(roi([rnd.choice(b) for _ in b]) for _ in range(n))
    return rs[int(0.05 * n)], rs[int(0.95 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/api_tennis_matches.csv")
    ap.add_argument("--odds", default="data/odds_history.csv")
    ap.add_argument("--params", default="model_params.json")
    ap.add_argument("--json", default=None, help="also write the report here")
    ap.add_argument("--write", action="store_true", help="save the blend refitted on all odds to --params")
    args = ap.parse_args()
    log = lambda s="": print(s, file=sys.stderr)

    with open(args.params) as fh:
        P = model.Params.from_dict(json.load(fh)["params"])
    data = bt.load(args.cache)
    cache = update_data.load_api_cache(args.cache)
    rows = joined(predictions(P, data, model.SetTable()), args.odds, cache)
    rows = [r for r in rows if r["date"] > bt.WARM_END]
    tune = [r for r in rows if bt.TUNE[0] <= r["date"] <= bt.TUNE[1]]
    hold = [r for r in rows if r["date"] >= bt.HOLD_START]
    log(f"{len(rows)} finished matches with odds after the warm-up: tuning period {len(tune)}, holdout {len(hold)}")
    report = {"n": {"tune": len(tune), "holdout": len(hold)}}

    # 1. sharpness
    report["logloss"] = {}
    for label, rs in (("tune", tune), ("holdout", hold)):
        out = {}
        for key in ("model", "market", "pin"):
            out[key] = logloss(rs, key)
        both = [r for r in rs if r["pin"] is not None]
        out["model_where_pin"] = logloss(both, "model")
        report["logloss"][label] = out
        log(f"{label:8} log loss: model {out['model'][0]:.4f}, market {out['market'][0]:.4f} (n={out['market'][1]})"
            + (f"; where Pinnacle priced it: model {out['model_where_pin'][0]:.4f}, Pinnacle {out['pin'][0]:.4f} "
               f"(n={out['pin'][1]})" if out["pin"][1] else ""))

    # 2. information: blend fitted on the tuning period, per level
    report["blend"] = {}
    for level in ("tour", "challenger"):
        tr = [r for r in tune if r["level"] == level]
        ho = [r for r in hold if r["level"] == level]
        if len(tr) < 200 or not ho:
            continue
        w = bt.fit_logistic([([L(r["market"]), L(r["model"])], r["y"]) for r in tr])
        for r in tune + hold:
            if r["level"] == level:
                r["blend"] = model.stack_prob(w, [L(r["market"]), L(r["model"])])
        hm, hb = logloss(ho, "market")[0], logloss(ho, "blend")[0]
        report["blend"][level] = {"weights": {"market": w[0], "model": w[1]}, "holdout_market": hm,
                                  "holdout_blend": hb, "n_holdout": len(ho)}
        log(f"blend {level:10}: market weight {w[0]:+.3f}, model weight {w[1]:+.3f}; holdout log loss "
            f"market {hm:.4f} -> blend {hb:.4f} (n={len(ho)})")

    # 3. betting on the holdout
    report["bets"] = {}
    segs = {"all": lambda r: True, "ATP": lambda r: r["tour"] == "ATP", "WTA": lambda r: r["tour"] == "WTA",
            "tour level": lambda r: r["level"] == "tour", "Challenger": lambda r: r["level"] == "challenger"}
    for key in ("model", "blend"):
        rs = [r for r in hold if key in r]
        for thr in THRESHOLDS:
            for seg, f in segs.items():
                b = bets([r for r in rs if f(r)], key, thr)
                if not b:
                    continue
                lo, hi = ci(b)
                report["bets"][f"{key} >{thr:.0%} {seg}"] = {"n": len(b), "roi": roi(b), "ci90": [lo, hi]}
                if seg == "all" or thr == 0.03:
                    log(f"  bet on {key:5} edge >{thr:4.0%} {seg:10}: {len(b):5} bets, ROI {roi(b):+.3f} "
                        f"[90% {lo:+.3f}, {hi:+.3f}]")
        for lo_, hi_ in ((1.0, 1.5), (1.5, 2.2), (2.2, 3.5), (3.5, 100)):
            b = [x for x in bets(rs, key, 0.03) if lo_ <= x[0] < hi_]
            if b:
                lo, hi = ci(b)
                report["bets"][f"{key} >3% price {lo_}-{hi_}"] = {"n": len(b), "roi": roi(b), "ci90": [lo, hi]}
                log(f"  bet on {key:5} edge > 3% price {lo_}-{hi_}: {len(b):5} bets, ROI {roi(b):+.3f} "
                    f"[90% {lo:+.3f}, {hi:+.3f}]")
    if args.write:
        weights = {}
        for level in ("tour", "challenger"):
            rs = [r for r in rows if r["level"] == level]
            if len(rs) >= 500:
                w = bt.fit_logistic([([L(r["market"]), L(r["model"])], r["y"]) for r in rs])
                weights[level] = [round(x, 4) for x in w]
                log(f"production blend {level}: market {w[0]:+.3f}, model {w[1]:+.3f} (n={len(rs)})")
        with open(args.params) as fh:
            saved = json.load(fh)
        saved["market_blend"] = {"weights": weights, "fitted_on": f"{min(r['date'] for r in rows)}.."
                                 f"{max(r['date'] for r in rows)}", "holdout": report["blend"],
                                 "note": "value bets use bet_prob(model, market) with these weights; "
                                         "see tools/market_backtest.py"}
        with open(args.params, "w") as fh:
            json.dump(saved, fh, indent=2, default=str)
        log(f"wrote market_blend to {args.params}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=1, default=str)
        log(f"wrote {args.json}")


if __name__ == "__main__":
    main()
