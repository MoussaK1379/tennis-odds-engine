"""Historical bookmaker odds for the market backtest (tools/market_backtest.py).

tennis-data.co.uk publishes a spreadsheet per tour and season with every ATP/WTA
tour-level match and its closing odds: Pinnacle (PSW/PSL), the best price across
the books it follows (MaxW/MaxL), their average (AvgW/AvgL) and Bet365 (B365W/B365L).
This saves the odds columns to data/odds_history.csv, names converted to the
"J. Sinner" form the api-tennis cache uses. Run weekly by .github/workflows/odds-history.yml.

    python tools/fetch_odds.py                  # 2024 to this season
    python tools/fetch_odds.py --years 2022 2023 2024 2025 2026
"""

import argparse, csv, os, sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import update_data as u                           # noqa: E402

FIELDS = ["tour", "date", "tournament", "surface", "round", "best_of", "winner", "loser",
          "w_rank", "l_rank", "comment", "ps_w", "ps_l", "max_w", "max_l", "avg_w", "avg_l",
          "b365_w", "b365_l"]
COLS = {"ps_w": "PSW", "ps_l": "PSL", "max_w": "MaxW", "max_l": "MaxL", "avg_w": "AvgW",
        "avg_l": "AvgL", "b365_w": "B365W", "b365_l": "B365L", "w_rank": "WRank", "l_rank": "LRank"}


def download(url):
    """The site answers 403 to the updater's bot user agent, so ask like a browser (https first)."""
    import urllib.request
    err = None
    for u_ in (url.replace("http://", "https://"), url):
        req = urllib.request.Request(u_, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
                                                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
                                                  "Accept": "*/*"})
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                return resp.read()
        except Exception as e:                       # noqa: BLE001
            err = e
    raise err


def rows_for(tour, year):
    out = []
    for r in u.read_xlsx(download(u.TENNIS_DATA[tour].format(year=year))):
        d = u.td_date(r.get("Date"))
        w, l = u.td_name(r.get("Winner")), u.td_name(r.get("Loser"))
        if not d or not w or not l:
            continue
        row = {"tour": tour.upper(), "date": d.isoformat(), "tournament": r.get("Tournament", ""),
               "surface": u.norm_surface(r.get("Surface")) or "", "round": r.get("Round", ""),
               "best_of": r.get("Best of", ""), "winner": w, "loser": l, "comment": r.get("Comment", "")}
        for k, c in COLS.items():
            v = u.to_float(r.get(c))
            row[k] = "" if v is None else v
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    yr = date.today().year
    ap.add_argument("--years", nargs="+", type=int, default=list(range(2024, yr + 1)))
    ap.add_argument("--out", default="data/odds_history.csv")
    args = ap.parse_args()
    old = []
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            old = list(csv.DictReader(fh))
    got, fetched = [], set()
    for tour in ("atp", "wta"):
        for y in args.years:
            try:
                rows = rows_for(tour, y)
            except Exception as e:                   # noqa: BLE001
                print(f"  · {tour} {y}: not available ({e})", file=sys.stderr)
                continue
            print(f"  · {tour} {y}: {len(rows)} matches", file=sys.stderr)
            got += rows
            fetched.add((tour.upper(), str(y)))
    if not got:
        sys.exit("no odds fetched; keeping the existing file")
    # seasons that failed this time keep their previous rows
    keep = [r for r in old if (r["tour"], r["date"][:4]) not in fetched]
    rows = sorted(keep + got, key=lambda r: (r["date"], r["tour"], r["tournament"], r["winner"]))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {args.out}: {len(rows)} matches", file=sys.stderr)


if __name__ == "__main__":
    main()
