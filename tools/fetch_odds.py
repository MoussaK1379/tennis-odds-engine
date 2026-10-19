"""Historical pre-match odds for the market backtest (tools/market_backtest.py).

api-tennis.com's get_odds takes past dates too, keyed by the same event_key as the
match cache (data/api_tennis_matches.csv). For every finished tour-level and
Challenger match with match-winner prices this saves, from the first player's side:
the best price, the average de-vigged probability across books, Pinnacle's de-vigged
probability and the number of books, to data/odds_history.csv. get_fixtures for the
same window says who the first player was (their key), so prices match the winner.

Each run fetches the last week, then works backwards to the start of the match cache
until the time budget runs out (progress in data/odds_history_state.json). Run daily
by .github/workflows/odds-history.yml (needs TENNIS_API_KEY).

    TENNIS_API_KEY=... python tools/fetch_odds.py [--budget 900]
"""

import argparse, csv, json, os, statistics, sys, time
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import update_data as u                           # noqa: E402

OUT = "data/odds_history.csv"
STATE = "data/odds_history_state.json"
FIELDS = ["event_key", "date", "tour", "first_key", "books", "best_1", "best_2", "fair_1", "pin_1"]
WINDOW = 3                    # days per request
REFRESH = 7                   # recent days re-fetched every run
WIN_MARKETS = ("home/away", "match winner", "to win match", "winner", "1x2")
EXCHANGES = ("exchange", "betfair", "matchbook", "smarkets", "betdaq")


def books_of(mk):
    """{bookmaker: (price first player, price second player)} from one event's markets."""
    win = next((mk[n] for n in mk if n.strip().lower() in WIN_MARKETS), None) if isinstance(mk, dict) else None
    if not isinstance(win, dict):
        return {}
    sides = {k.strip().lower(): v for k, v in win.items() if isinstance(v, dict)}
    home, away = sides.get("home") or sides.get("1"), sides.get("away") or sides.get("2")
    out = {}
    for bk in home or {}:
        try:
            pa, pb = float(home[bk]), float((away or {}).get(bk))
        except (TypeError, ValueError):
            continue
        if pa > 1 and pb > 1 and 1 / pa + 1 / pb < 1.25:      # drop obviously broken quotes
            out[bk] = (pa, pb)
    return out


def summarise(books):
    fair, pin = [], ""
    for bk, (pa, pb) in books.items():
        f = (1 / pa) / (1 / pa + 1 / pb)
        fair.append(f)
        if "pinnacle" in bk.lower() or bk.lower() == "pncl":   # api-tennis abbreviates it "Pncl"
            pin = round(f, 4)
    soft = {bk: v for bk, v in books.items() if not any(x in bk.lower() for x in EXCHANGES)} or books
    return {"books": len(books), "best_1": max(v[0] for v in soft.values()),
            "best_2": max(v[1] for v in soft.values()), "fair_1": round(statistics.mean(fair), 4), "pin_1": pin}


def fetch_window(key, a, b):
    """-> (rows for finished matches between a and b with prices, finished matches, events with odds)."""
    first = {}
    for tour, (type_key, type_name) in u.API_TENNIS_TYPES.items():
        for ev in u.api_tennis_call(key, method="get_fixtures", event_type_key=type_key,
                                    date_start=a.isoformat(), date_stop=b.isoformat()):
            if ev.get("event_type_type") == type_name and ev.get("event_status") in u.API_STATUS:
                first[str(ev.get("event_key"))] = (tour.upper(), ev.get("event_date"), ev.get("first_player_key"))
    res = u.api_tennis_call(key, method="get_odds", date_start=a.isoformat(), date_stop=b.isoformat())
    res = res if isinstance(res, dict) else {}
    rows = []
    for ek, mk in res.items():
        if str(ek) not in first:
            continue
        books = books_of(mk)
        if books:
            tour, d, fk = first[str(ek)]
            rows.append({"event_key": str(ek), "date": d, "tour": tour, "first_key": fk, **summarise(books)})
    return rows, len(first), len(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=900, help="seconds to spend fetching")
    ap.add_argument("--cache", default="data/api_tennis_matches.csv")
    args = ap.parse_args()
    key = os.environ.get("TENNIS_API_KEY", "").strip()
    if not key:
        sys.exit("TENNIS_API_KEY not set")
    t0 = time.time()
    rows = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as fh:
            rows = {r["event_key"]: r for r in csv.DictReader(fh)}
    state = u.load_json(STATE) or {}
    start = datetime.strptime(min(r["date"] for r in u.load_api_cache(args.cache).values()), "%Y-%m-%d").date()
    today = date.today()

    def save():
        with open(OUT, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=FIELDS)
            wr.writeheader()
            wr.writerows(sorted(rows.values(), key=lambda r: (r["date"], r["event_key"])))
        with open(STATE, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=1)

    def run(a, b):
        got, n_fx, n_odds = fetch_window(key, a, b)
        for r in got:
            rows[r["event_key"]] = r
        print(f"  · {a}..{b}: {n_fx} finished matches, odds for {n_odds} events, {len(got)} matched",
              file=sys.stderr)

    calls = 0
    try:
        # recent days first (late results, closing prices), then history backwards
        a = today - timedelta(days=REFRESH)
        while a < today:
            b = min(a + timedelta(days=WINDOW - 1), today - timedelta(days=1))
            run(a, b)
            a = b + timedelta(days=1)
        b = (datetime.strptime(state["from"], "%Y-%m-%d").date() - timedelta(days=1) if state.get("from")
             else today - timedelta(days=REFRESH + 1))
        while b >= start and time.time() - t0 < args.budget:
            a = max(b - timedelta(days=WINDOW - 1), start)
            run(a, b)
            state["from"] = a.isoformat()
            b = a - timedelta(days=1)
            calls += 1
            if calls % 10 == 0:
                save()                               # a killed run keeps its work
    except Exception as e:                           # noqa: BLE001
        print(f"  · stopped: {str(e).replace(key, '***')[:300]}", file=sys.stderr)
    save()
    print(f"wrote {OUT}: {len(rows)} matches with odds; history back to {state.get('from')} "
          f"(cache starts {start})", file=sys.stderr)


if __name__ == "__main__":
    main()
