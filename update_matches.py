"""Daily matches + odds updater for the Tennis Odds Engine.

Pulls upcoming ATP and WTA matches with bookmaker odds from The Odds API
(https://the-odds-api.com), runs every match through the same model the web app
uses (ELO + Markov serve model, 50/50 ensemble), and writes matches.json with
the model probability, the best available price, expected value and a
quarter-Kelly stake for each side. The web app shows it as "Today's matches".

Needs a free API key in the ODDS_API_KEY environment variable. Without one it
writes an empty matches.json that says so, and exits cleanly so the rest of
the daily job still runs.

Self-contained (standard library only) so it runs cleanly on a CI runner.

Usage:
    ODDS_API_KEY=... python update_matches.py
    ODDS_API_KEY=... python update_matches.py --hours 48 --regions eu,uk
"""

import argparse, json, os, re, statistics, sys, unicodedata
import urllib.parse, urllib.request
from datetime import datetime, timedelta, timezone
from functools import lru_cache

API = "https://api.the-odds-api.com/v4"

GRAND_SLAMS = ("australian open", "french open", "roland garros", "wimbledon", "us open")

# Tournament surfaces. Titles are normalised to lowercase ASCII words first
# ("Båstad" -> "bastad", "Monte-Carlo" -> "monte carlo"), then the first
# matching row wins, so more specific names sit above looser ones.
# Each row: (name keywords, surface, indoor). The model has no indoor
# adjustment; indoor is shown on the site for information.
TOURNAMENTS = [
    # Grand Slams
    (("australian open",), "hard", False),
    (("french open", "roland garros"), "clay", False),
    (("wimbledon",), "grass", False),
    (("us open",), "hard", False),
    # Year-end and team events
    (("next gen",), "hard", False),                   # before "atp finals"
    (("atp finals", "nitto", "turin"), "hard", True),
    (("united cup",), "hard", False),
    (("laver cup",), "hard", True),
    (("davis cup",), "hard", True),
    # Masters 1000
    (("indian wells", "bnp paribas open"), "hard", False),
    (("miami",), "hard", False),
    (("monte carlo", "monaco"), "clay", False),
    (("madrid",), "clay", False),
    (("rome", "italian open", "internazionali"), "clay", False),
    (("montreal", "toronto", "canadian open", "national bank open"), "hard", False),
    (("cincinnati",), "hard", False),
    (("shanghai",), "hard", False),
    (("paris masters", "rolex paris", "paris"), "hard", True),
    # ATP 500
    (("dallas",), "hard", True),
    (("rotterdam",), "hard", True),
    (("doha", "qatar"), "hard", False),
    (("rio de janeiro", "rio open"), "clay", False),
    (("acapulco", "mexican open"), "hard", False),
    (("dubai",), "hard", False),
    (("barcelona",), "clay", False),
    (("munich",), "clay", False),
    (("hamburg",), "clay", False),
    (("halle",), "grass", False),
    (("queen s", "queens"), "grass", False),
    (("washington", "citi open"), "hard", False),
    (("tokyo", "japan open"), "hard", False),
    (("beijing", "china open"), "hard", False),
    (("basel", "swiss indoors"), "hard", True),
    (("vienna", "erste bank"), "hard", True),
    # ATP 250
    (("brisbane",), "hard", False),
    (("hong kong",), "hard", False),
    (("adelaide",), "hard", False),
    (("auckland", "asb classic"), "hard", False),
    (("montpellier", "open occitanie"), "hard", True),
    (("buenos aires", "argentina open"), "clay", False),
    (("delray beach",), "hard", False),
    (("santiago", "chile open"), "clay", False),
    (("bucharest",), "clay", False),
    (("houston", "clay court"), "clay", False),
    (("marrakech", "morocco"), "clay", False),
    (("geneva",), "clay", False),
    (("s hertogenbosch", "hertogenbosch", "libema"), "grass", False),
    (("mallorca",), "grass", False),
    (("eastbourne",), "grass", False),
    (("bastad", "swedish open"), "clay", False),
    (("gstaad", "swiss open"), "clay", False),
    (("umag", "croatia open"), "clay", False),
    (("kitzbuhel", "austrian open"), "clay", False),
    (("estoril",), "clay", False),
    (("los cabos",), "hard", False),
    (("winston salem",), "hard", False),
    (("chengdu",), "hard", False),
    (("hangzhou",), "hard", False),
    (("almaty",), "hard", True),
    (("brussels", "antwerp", "european open"), "hard", True),
    (("lyon",), "hard", True),
    (("stockholm", "nordic open"), "hard", True),
]
# Stuttgart is grass on the ATP tour but clay (indoor) on the WTA tour.
TOUR_SPECIFIC = {
    ("ATP", "stuttgart"): ("grass", False),
    ("WTA", "stuttgart"): ("clay", True),
}
# WTA-only events not on the ATP list above.
WTA_EXTRA = [
    (("charleston",), "clay", False),
    (("berlin", "bad homburg", "birmingham", "nottingham"), "grass", False),
    (("wuhan", "ningbo", "guadalajara", "seoul", "osaka", "abu dhabi",
      "linz", "ostrava", "san diego", "cleveland", "monterrey"), "hard", False),
    (("rabat", "strasbourg", "palermo", "prague", "bogota"), "clay", False),
]


# ---------- engine (mirrors the JS in index.html) -----------------------------
def clip(p, lo=0.01, hi=0.99): return max(lo, min(hi, p))
def elo_expected(ra, rb): return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
def serve_point_prob(s, r, t): return clip(t + (s - t) - (r - (1 - t)))

def game_win_prob(p):
    q = 1 - p
    den = p * p + q * q
    pd = (p * p) / den if den > 0 else 0.5
    return p**4 + 4 * p**4 * q + 10 * p**4 * q * q + 20 * p**3 * q**3 * pd

def _tb_server_is_a(a, b): return True if a + b == 0 else ((a + b - 1) // 2) % 2 == 1

def tiebreak_win_prob(pa, pb):
    a_s, a_r = pa, 1 - pb
    den = a_s * a_r + (1 - a_s) * (1 - a_r)
    tail = (a_s * a_r) / den if den > 0 else 0.5

    @lru_cache(maxsize=None)
    def f(a, b):
        if a == b and a >= 6: return tail
        if a >= 7 and a - b >= 2: return 1.0
        if b >= 7 and b - a >= 2: return 0.0
        ppt = pa if _tb_server_is_a(a, b) else 1 - pb
        return ppt * f(a + 1, b) + (1 - ppt) * f(a, b + 1)
    return f(0, 0)

def set_win_prob(pa, pb):
    ga, gb, tb = game_win_prob(pa), game_win_prob(pb), tiebreak_win_prob(pa, pb)

    @lru_cache(maxsize=None)
    def f(a, b, a_serves):
        if a == 6 and b == 6: return tb
        if a >= 6 and a - b >= 2: return 1.0
        if b >= 6 and b - a >= 2: return 0.0
        pg = ga if a_serves else 1 - gb
        return pg * f(a + 1, b, not a_serves) + (1 - pg) * f(a, b + 1, not a_serves)
    return 0.5 * f(0, 0, True) + 0.5 * f(0, 0, False)

def match_from_set(s, best_of):
    return s * s * (3 - 2 * s) if best_of == 3 else s**3 * (6 * s * s - 15 * s + 10)

def kelly_fraction(p, d, frac=0.25):
    b = d - 1
    return max(0.0, frac * ((b * p - (1 - p)) / b)) if b > 0 else 0.0


def _pick(obj, surf):
    if not obj: return None
    return obj[surf] if obj.get(surf) is not None else obj.get("overall")

def player_stats(p, surf):
    return {"elo": _pick(p["elo"], surf),
            "spw": clip(_pick(p["serve"], surf), 0.50, 0.85),
            "rpw": clip(_pick(p["return"], surf), 0.20, 0.55)}

def predict(pa_data, pb_data, surf, best_of, tour_avg):
    A, B = player_stats(pa_data, surf), player_stats(pb_data, surf)
    pa = serve_point_prob(A["spw"], B["rpw"], tour_avg)
    pb = serve_point_prob(B["spw"], A["rpw"], tour_avg)
    p_mkv = match_from_set(set_win_prob(pa, pb), best_of)
    p_elo = elo_expected(A["elo"], B["elo"])
    return 0.5 * (p_elo + p_mkv)


# ---------- helpers -----------------------------------------------------------
def slugify(name):
    n = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", n.lower()).strip("-")

def build_name_index(players):
    """slug -> key, plus 'first-initial + surname' -> key when that's unambiguous."""
    exact, loose, clashes = {}, {}, set()
    for key, p in players.items():
        s = slugify(p["name"])
        exact[s] = key
        parts = s.split("-")
        if len(parts) >= 2:
            lk = parts[0][0] + "-" + "-".join(parts[1:])
            if lk in loose and loose[lk] != key: clashes.add(lk)
            loose[lk] = key
    for lk in clashes: loose.pop(lk, None)
    return exact, loose

def resolve(name, tour, players, idx):
    exact, loose = idx
    s = slugify(name)
    cands = [exact.get(s)]
    parts = s.split("-")
    if len(parts) >= 2:
        cands.append(loose.get(parts[0][0] + "-" + "-".join(parts[1:])))
        cands.append(exact.get("-".join(parts[1:] + parts[:1])))    # "Zheng Qinwen" vs "Qinwen Zheng"
    for k in cands:
        if k and players[k].get("tour", "ATP") == tour:
            return k
    return None

def _words(title):
    t = unicodedata.normalize("NFKD", title or "").encode("ascii", "ignore").decode().lower()
    return " " + re.sub(r"[^a-z0-9]+", " ", t).strip() + " "

def surface_for(title, tour):
    """-> (surface, indoor, known). Unknown tournaments default to outdoor hard."""
    t = _words(title)
    has = lambda k: f" {k} " in t
    for (tr, k), (surf, indoor) in TOUR_SPECIFIC.items():
        if tr == tour and has(k):
            return surf, indoor, True
    for keys, surf, indoor in TOURNAMENTS + (WTA_EXTRA if tour == "WTA" else []):
        if any(has(k) for k in keys):
            return surf, indoor, True
    return "hard", False, False

def guess_best_of(title, tour):
    t = _words(title)
    return 5 if tour == "ATP" and any(f" {k} " in t for k in GRAND_SLAMS) else 3

def iso(dt): return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def get_json(path, params):
    url = f"{API}{path}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp), resp.headers.get("x-requests-remaining")


# ---------- odds --------------------------------------------------------------
def price_summary(event, a, b):
    """Best price per side, plus the de-vigged market probability for A
    (average of each bookmaker's own de-vigged number)."""
    best = {a: (0.0, None), b: (0.0, None)}
    fair_a = []
    for bk in event.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m.get("key") != "h2h": continue
            prices = {o["name"]: o.get("price") for o in m.get("outcomes", [])}
            pa_, pb_ = prices.get(a), prices.get(b)
            if not (pa_ and pb_ and pa_ > 1 and pb_ > 1): continue
            if pa_ > best[a][0]: best[a] = (pa_, bk.get("title"))
            if pb_ > best[b][0]: best[b] = (pb_, bk.get("title"))
            ia, ib = 1 / pa_, 1 / pb_
            fair_a.append(ia / (ia + ib))
    if not fair_a: return None
    return best, statistics.mean(fair_a), len(fair_a)

def side(model_p, market_p, price, book):
    ev = model_p * price - 1
    return {"model": round(model_p, 4), "fair_odds": round(1 / model_p, 2),
            "market": round(market_p, 4), "best_odds": price, "book": book,
            "ev": round(ev, 4), "kelly": round(kelly_fraction(model_p, price), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", default="players.json")
    ap.add_argument("--out", default="matches.json")
    ap.add_argument("--hours", type=int, default=36, help="look this far ahead")
    ap.add_argument("--regions", default="eu", help="bookmaker regions: us, uk, eu, au (each costs 1 request)")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    out = {"updated": iso(now), "window_hours": args.hours, "status": "ok",
           "requests_remaining": None, "matches": [], "unmatched": []}

    def write():
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)

    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        out["status"] = "no_api_key"
        write()
        print("ODDS_API_KEY not set — wrote an empty matches.json.", file=sys.stderr)
        return

    with open(args.players, encoding="utf-8") as fh:
        data = json.load(fh)
    players = data["players"]
    idx = build_name_index(players)
    tour_avgs = {t.upper(): v.get("tour_avg_spw") for t, v in (data.get("tours") or {}).items()}

    sports, remaining = get_json("/sports", {"apiKey": key})       # this call is free
    tennis = [s for s in sports if s.get("active") and re.match(r"tennis_(atp|wta)_", s.get("key", ""))]
    print(f"{len(tennis)} active ATP/WTA events", file=sys.stderr)

    for sp in tennis:
        tour = "ATP" if sp["key"].startswith("tennis_atp") else "WTA"
        title = sp.get("title", sp["key"])
        try:
            events, remaining = get_json(f"/sports/{sp['key']}/odds", {
                "apiKey": key, "regions": args.regions, "markets": "h2h", "oddsFormat": "decimal",
                "commenceTimeFrom": iso(now), "commenceTimeTo": iso(now + timedelta(hours=args.hours)),
            })
        except Exception as e:                                        # noqa: BLE001
            print(f"  · {title}: odds unavailable ({e})", file=sys.stderr)
            continue

        surf, indoor, known = surface_for(title, tour)
        best_of = guess_best_of(title, tour)
        if not known:
            print(f"  · {title}: surface unknown, assuming hard", file=sys.stderr)
        t_avg = tour_avgs.get(tour) or (0.64 if tour == "ATP" else 0.56)
        for ev in events:
            a, b = ev.get("home_team"), ev.get("away_team")
            ka, kb = resolve(a, tour, players, idx), resolve(b, tour, players, idx)
            summary = price_summary(ev, a, b)
            row = {"id": ev.get("id"), "tour": tour, "tournament": title,
                   "start": ev.get("commence_time"), "surface": surf, "indoor": indoor,
                   "surface_known": known, "best_of": best_of,
                   "a": {"name": a, "key": ka}, "b": {"name": b, "key": kb}}
            if not summary:
                continue
            (best, mkt_a, n_books) = summary
            row["books"] = n_books
            if not (ka and kb):
                row["missing"] = [n for n, k in ((a, ka), (b, kb)) if not k]
                out["unmatched"].append(row)
                continue
            p = predict(players[ka], players[kb], surf, best_of, t_avg)
            row["a"].update(side(p, mkt_a, *best[a]))
            row["b"].update(side(1 - p, 1 - mkt_a, *best[b]))
            out["matches"].append(row)

    out["matches"].sort(key=lambda r: r["start"] or "")
    out["unmatched"].sort(key=lambda r: r["start"] or "")
    out["requests_remaining"] = remaining
    write()
    n_val = sum(1 for m in out["matches"] if m["a"]["ev"] > 0 or m["b"]["ev"] > 0)
    print(f"Wrote {args.out}: {len(out['matches'])} matches ({n_val} with a +EV side), "
          f"{len(out['unmatched'])} unmatched, requests remaining: {remaining}", file=sys.stderr)


if __name__ == "__main__":
    main()
