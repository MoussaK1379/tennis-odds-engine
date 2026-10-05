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

import argparse, csv, json, math, os, re, statistics, sys, unicodedata
import urllib.parse, urllib.request
from datetime import datetime, timedelta, timezone

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
    (("wta finals", "riyadh"), "hard", True),
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
    (("london",), "grass", False),                    # api-tennis calls Queen's "London"; after Laver Cup/Finals
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
    (("marseille", "open 13"), "hard", True),
    (("metz", "moselle"), "hard", True),
    (("athens", "hellenic"), "hard", True),
    (("belgrade",), "hard", True),
]
# Stuttgart is grass on the ATP tour but clay (indoor) on the WTA tour.
TOUR_SPECIFIC = {
    ("ATP", "stuttgart"): ("grass", False),
    ("WTA", "stuttgart"): ("clay", True),
}
# Events whose surface depends on the time of year: (tour, keyword) -> {month: (surface, indoor)}.
# Linz moved from indoor hard (Jan/Feb) to indoor clay (April) in 2026.
SEASONAL = {
    ("WTA", "linz"): {3: ("clay", True), 4: ("clay", True), 5: ("clay", True)},
}
# WTA-only events not on the ATP list above. Each keyword here is its own court.
WTA_EXTRA = [
    (("charleston",), "clay", False),
    (("berlin", "bad homburg", "birmingham", "nottingham"), "grass", False),
    (("linz", "ostrava", "cluj", "singapore"), "hard", True),
    (("wuhan", "ningbo", "guadalajara", "seoul", "osaka", "abu dhabi", "san diego",
      "cleveland", "monterrey", "austin", "chennai", "guangzhou", "hobart", "jiujiang",
      "merida", "sao paulo", "prague"), "hard", False),
    (("rouen",), "clay", True),
    (("rabat", "strasbourg", "palermo", "bogota", "iasi"), "clay", False),
]


# ---------- model (shared with update_data.py and the backtest) ----------------
from model import predict_export  # noqa: E402  (after the tables above, which model-free callers import)

def kelly_fraction(p, d, frac=0.25):
    b = d - 1
    return max(0.0, frac * ((b * p - (1 - p)) / b)) if b > 0 else 0.0


# ---------- helpers -----------------------------------------------------------
def slugify(name):
    n = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", n.lower()).strip("-")

def name_tiers(name):
    """Keys an odds-feed name could appear under in the stats feed, strictest first.

    The stats feed writes 'J. Sinner', 'T. M. Etcheverry', 'C. Osorio', 'Y. Bu';
    the odds feed writes full names in either order ('Bu Yunchaokete',
    'Maria Camila Osorio Serrano')."""
    t = [w for w in slugify(name).split("-") if w]
    if len(t) < 2:
        return [{"-".join(t)}]
    last = t[-1]
    # Western order is tried before surname-first order at each step, so
    # 'Maria Timofeeva' finds M. Timofeeva before T. Maria (Tatjana Maria).
    return [
        {"-".join(t)},                                                   # exact
        {"-".join(t[1:] + t[:1])},                                       # exact, reversed
        {t[0][0] + "-" + "-".join(t[1:])},                               # J. Sinner
        {"-".join(w[0] for w in t[:-1]) + "-" + last},                   # T. M. Etcheverry
        {t[-1][0] + "-" + "-".join(t[:-1])},                             # Y. Bu for 'Bu Yunchaokete'
        {t[i][0] + "-" + t[j] for i in range(len(t)) for j in range(i + 1, len(t))},  # C. Osorio, G. Ruse
        {t[i][0] + "-" + t[j] for i in range(len(t)) for j in range(i)},              # surname-first pairs
    ]

def _player_keys(name):
    parts = [w for w in slugify(name).split("-") if w]
    return {"-".join(parts), parts[0][0] + "-" + "-".join(parts[1:])} if len(parts) >= 2 else {"-".join(parts)}

def build_name_index(players):
    """tour -> {name key -> set of player ids}."""
    idx = {}
    for pid, p in players.items():
        m = idx.setdefault(p.get("tour", "ATP"), {})
        for k in _player_keys(p["name"]):
            m.setdefault(k, set()).add(pid)
    return idx

def resolve(name, tour, players, idx):
    """Player id for an odds-feed name, or None. A looser tier is used only if
    the stricter ones found nobody, and only when it points at exactly one player."""
    m = idx.get(tour, {})
    for tier in name_tiers(name):
        hits = set().union(*(m.get(k, set()) for k in tier))
        if len(hits) == 1:
            return hits.pop()
        if len(hits) > 1:
            return None          # ambiguous: better unmatched than wrong
    return None

def _words(title):
    t = unicodedata.normalize("NFKD", title or "").encode("ascii", "ignore").decode().lower()
    return " " + re.sub(r"[^a-z0-9]+", " ", t).strip() + " "

def _match_event(title, tour, month=None):
    """-> (court id, surface, indoor) or None. The court id is the same for a
    tournament however each feed names it ('Tokyo' / 'ATP Japan Open' -> 'tokyo')."""
    t = _words(title)
    has = lambda k: f" {k} " in t
    for (tr, k), by_month in SEASONAL.items():
        if tr == tour and has(k) and month in by_month:
            return (k,) + by_month[month]
    for (tr, k), (surf, indoor) in TOUR_SPECIFIC.items():
        if tr == tour and has(k):
            return k, surf, indoor
    for keys, surf, indoor in TOURNAMENTS:
        if any(has(k) for k in keys):
            return keys[0], surf, indoor
    if tour == "WTA":
        for keys, surf, indoor in WTA_EXTRA:
            for k in keys:
                if has(k):
                    return k, surf, indoor
    return None

def surface_for(title, tour, month=None):
    """-> (surface, indoor, known). Unknown tournaments default to outdoor hard."""
    hit = _match_event(title, tour, month)
    return (hit[1], hit[2], True) if hit else ("hard", False, False)

def court_id(title, tour, month=None):
    """Stable id for a tournament's court, shared by both feeds; None if unknown."""
    hit = _match_event(title, tour, month)
    return hit[0] if hit else None

def guess_best_of(title, tour):
    t = _words(title)
    return 5 if tour == "ATP" and any(f" {k} " in t for k in GRAND_SLAMS) else 3

def iso(dt): return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def get_json(path, params):
    url = f"{API}{path}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp), resp.headers.get("x-requests-remaining")


# ---------- odds --------------------------------------------------------------
# Betting exchanges: their prices exclude the commission they take on winnings,
# and they don't offer parlays, so they're left out of parlay pricing.
EXCHANGES = ("matchbook", "smarkets", "betdaq")

def is_exchange(key):
    return "_ex_" in key or key in EXCHANGES

def price_summary(event, a, b):
    """Best price per side, the de-vigged market probability for A (average of
    each bookmaker's own de-vigged number), the number of books, and Pinnacle's
    de-vigged probability for A when Pinnacle prices the match, and every
    sportsbook's prices ({book: [price A, price B]}, exchanges left out) for
    building parlays."""
    best = {a: (0.0, None), b: (0.0, None)}
    fair_a, pinnacle_a, by_book = [], None, {}
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
            if not is_exchange(bk.get("key", "")):      # exchanges don't take parlays
                by_book[bk.get("title") or bk.get("key")] = [pa_, pb_]
            if bk.get("key") == "pinnacle":           # the sharpest book: the real benchmark
                pinnacle_a = ia / (ia + ib)
    if not fair_a: return None
    return best, statistics.mean(fair_a), len(fair_a), pinnacle_a, by_book

# Warnings shown next to a match. Each is (code, text); the codes go in the log
# so the track record can be split by whether a pick carried warnings.
THIN_MATCHES = 10        # fewer matches than this in the last 12 months
LAYOFF_DAYS = 45         # longer than this since the last match
SURFACE_FEW = 3          # fewer matches than this on the surface in 12 months
SPECIALIST_SHARE = 0.5   # at least this share of the year's matches on clay or on grass
RECENT_RET_DAYS = 60
DISAGREE = 0.15          # model and market this far apart (15 percentage points)

def player_flags(p, surface, today):
    i, out, name = p.get("info") or {}, [], p.get("name", "?")
    if p.get("new"):
        return [("thin", f"{name}: no matches in our data yet, rated as a first-time qualifier")]
    if i.get("m12", 0) < THIN_MATCHES:
        out.append(("thin", f"{name}: only {i.get('m12', 0)} matches in 12 months"))
    if i.get("last"):
        gap = (today - datetime.strptime(i["last"], "%Y-%m-%d").date()).days
        if gap > LAYOFF_DAYS:
            out.append(("layoff", f"{name}: first match in {gap} days"))
    if (i.get("surf12") or {}).get(surface, 0) < SURFACE_FEW:
        out.append(("surface_few", f"{name}: few {surface} matches this year"))
    # Hard courts are most of the calendar, so only clay and grass make a specialist.
    m12, surf12 = i.get("m12", 0), i.get("surf12") or {}
    for spec in ("clay", "grass"):
        if m12 >= THIN_MATCHES and surf12.get(spec, 0) / m12 >= SPECIALIST_SHARE:
            share = f"{surf12[spec] / m12:.0%} of matches on {spec}"
            out.append(("specialist", f"{name}: {spec}-court specialist ({share})"
                        + ("" if spec == surface else f", now on {surface}")))
    if i.get("last_ret"):
        ago = (today - datetime.strptime(i["last_ret"], "%Y-%m-%d").date()).days
        if ago <= RECENT_RET_DAYS or i.get("ret180", 0) >= 2:
            out.append(("retired", f"{name}: retired from a match {ago} days ago"
                        + (f" ({i['ret180']} in 6 months)" if i.get("ret180", 0) >= 2 else "")))
    return out

def match_flags(A, B, surface, today, model_p, market_p):
    out = player_flags(A, surface, today) + player_flags(B, surface, today)
    if abs(model_p - market_p) >= DISAGREE:
        out.append(("disagree", f"Model and market differ by {abs(model_p - market_p) * 100:.0f} points"))
    return out

def side(model_p, market_p, price, book):
    ev = model_p * price - 1
    return {"model": round(model_p, 4), "fair_odds": round(1 / model_p, 2),
            "market": round(market_p, 4), "best_odds": price, "book": book,
            "ev": round(ev, 4), "kelly": round(kelly_fraction(model_p, price), 4)}


# ---------- parlays ---------------------------------------------------------------
# A parlay multiplies the bookmaker's margin and the model's errors, so legs are
# held to a stricter standard than single picks, and the whole ticket is priced
# at ONE bookmaker that offers every leg (prices can't be mixed across books).
PARLAY_LEG_MIN_EV = 0.03     # each leg must be value at the parlay's bookmaker
PARLAY_LEG_MIN_P = 0.40      # no long shots
PARLAY_MIN_EV = 0.05
PARLAY_SIZES = (2, 3)
PARLAY_SHOW = 3

def build_parlays(matches, now):
    from itertools import combinations
    books = {}
    for m in matches:
        st = parse_iso(m.get("start"))
        if not st or st <= now or m.get("flags"):
            continue
        for book, prices in (m.get("prices") or {}).items():
            for side, price in zip(("a", "b"), prices):
                p = m[side]["model"]
                if p >= PARLAY_LEG_MIN_P and p * price - 1 >= PARLAY_LEG_MIN_EV:
                    books.setdefault(book, []).append({
                        "id": m["id"], "side": side, "name": m[side]["name"],
                        "vs": m["b" if side == "a" else "a"]["name"], "tournament": m["tournament"],
                        "start": m["start"], "odds": price, "model": p, "market": m[side]["market"]})
    cands = []
    for book, legs in books.items():
        for k in PARLAY_SIZES:
            for combo in combinations(legs, k):
                if len({l["id"] for l in combo}) < k:
                    continue                       # both sides of one match
                odds = math.prod(l["odds"] for l in combo)
                p = math.prod(l["model"] for l in combo)       # matches are independent
                ev = p * odds - 1
                if ev >= PARLAY_MIN_EV:
                    cands.append({"book": book, "legs": sorted(combo, key=lambda l: l["start"]),
                                  "odds": round(odds, 2), "model": round(p, 4),
                                  "market": round(math.prod(l["market"] for l in combo), 4),
                                  "ev": round(ev, 4), "kelly": round(kelly_fraction(p, odds), 4)})
    # Best first, but keep the list varied: each new ticket must differ from the ones
    # already chosen by at least one leg, and no match appears in more than two.
    cands.sort(key=lambda c: (-c["ev"], -c["model"]))
    chosen, used = [], {}
    for c in cands:
        ids = frozenset(l["id"] for l in c["legs"])
        if any(ids == frozenset(l["id"] for l in x["legs"]) for x in chosen):
            continue
        if any(used.get(i, 0) >= 2 for i in ids):
            continue
        chosen.append(c)
        for i in ids:
            used[i] = used.get(i, 0) + 1
        if len(chosen) >= PARLAY_SHOW:
            break
    for c in chosen:
        c["id"] = "+".join(f"{l['id']}:{l['side']}" for l in c["legs"])
    return chosen

PARLAY_FIELDS = ["id", "created", "book", "legs", "odds", "model", "market", "ev", "result", "settled", "payout"]

def load_parlays(path):
    try:
        with open(path, encoding="utf-8", newline="") as fh:
            return {r["id"]: r for r in csv.DictReader(fh)}
    except OSError:
        return {}

def save_parlays(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PARLAY_FIELDS)
        w.writeheader()
        for r in sorted(rows.values(), key=lambda r: r["created"]):
            w.writerow({k: r.get(k, "") for k in PARLAY_FIELDS})

def log_parlays(rows, parlays, now):
    """First recommendation wins: a ticket's price is the one shown when it first appeared."""
    for c in parlays:
        if c["id"] not in rows:
            rows[c["id"]] = {"id": c["id"], "created": iso(now), "book": c["book"],
                             "legs": json.dumps([{k: l[k] for k in ("id", "side", "name", "odds")} for l in c["legs"]]),
                             "odds": c["odds"], "model": c["model"], "market": c["market"], "ev": c["ev"],
                             "result": "", "settled": "", "payout": ""}

def is_kalshi(r):
    return r.get("book") == "Kalshi"

def todays_parlays(rows, hist, matches, now, kalshi=False):
    """The day's recommended parlays, as logged. At most PARLAY_SHOW are logged per
    UTC day, so what the site shows is always exactly what the log holds."""
    by_id = {m["id"]: m for m in matches}
    out = []
    for r in sorted(rows.values(), key=lambda r: r["created"]):
        if r["created"][:10] != now.date().isoformat() or is_kalshi(r) != kalshi:
            continue
        legs = []
        for l in json.loads(r["legs"]):
            h, m = hist.get(l["id"]) or {}, by_id.get(l["id"])
            a_side = l["side"] == "a"
            legs.append({"id": l["id"], "side": l["side"], "name": l["name"], "odds": float(l["odds"]),
                         "vs": (m["b" if a_side else "a"]["name"] if m else h.get("player_b" if a_side else "player_a", "")),
                         "tournament": (m or h).get("tournament", ""), "start": (m or h).get("start", ""),
                         "url": ((m or {}).get("kalshi") or {}).get("url") if kalshi else None,
                         "model": (m[l["side"]]["model"] if m else
                                   (float(h["model_a"]) if a_side else 1 - float(h["model_a"])) if h.get("model_a") else 0)})
        odds, p = float(r["odds"]), float(r["model"])
        out.append({"id": r["id"], "book": r["book"], "legs": legs, "odds": odds, "model": p,
                    "market": float(r["market"]), "ev": float(r["ev"]), "kelly": round(kelly_fraction(p, odds), 4),
                    "created": r["created"], "result": r.get("result", "")})
    return out

def settle_parlays(rows, hist, now):
    """Settle from the single-match log. A void leg counts as odds 1.0 (usual bookmaker rule)."""
    for r in rows.values():
        if r.get("result"):
            continue
        legs = json.loads(r["legs"])
        res = [(hist.get(l["id"]) or {}).get("result") for l in legs]
        if any(x and x != "void" and x != l["side"] for x, l in zip(res, legs)):
            r["result"], r["payout"] = "lost", 0
        elif all(res):
            live = [l for x, l in zip(res, legs) if x != "void"]
            if not live:
                r["result"], r["payout"] = "void", 1
            else:
                r["result"], r["payout"] = "won", round(math.prod(float(l["odds"]) for l in live), 2)
        else:
            continue
        r["settled"] = iso(now)

def parlay_record(rows):
    done = [r for r in rows.values() if r.get("result") in ("won", "lost")]
    rec = {"settled": len(done), "pending": sum(1 for r in rows.values() if not r.get("result"))}
    if done:
        profit = sum(float(r["payout"]) - 1 for r in done)
        rec.update(won=sum(r["result"] == "won" for r in done), profit_units=round(profit, 2),
                   roi=round(profit / len(done), 4),
                   expected_wins=round(sum(float(r["model"]) for r in done), 1))
    return rec


# ---------- prediction log + track record ------------------------------------
HIST_FIELDS = ["id", "tour", "tournament", "start", "surface", "best_of", "player_a", "player_b",
               "model_a", "market_a", "odds_a", "odds_b", "books", "logged", "result", "settled",
               "pinnacle_a", "flags", "model_version"]
SETTLE_AFTER_H = 3        # try to settle once a match started this long ago
GIVE_UP_DAYS = 30         # after this, a match with no result found is voided

def loose_key(name):
    parts = slugify(name).split("-")
    return parts[0][:1] + "-" + "-".join(parts[1:]) if len(parts) >= 2 else slugify(name)

def name_keys(name):
    """Every key a name could match under (both feeds' styles, either order)."""
    return set().union(*name_tiers(name))

def parse_iso(s):
    try:
        return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None

def load_history(path):
    try:
        with open(path, encoding="utf-8", newline="") as fh:
            return {r["id"]: r for r in csv.DictReader(fh)}
    except OSError:
        return {}

def save_history(path, hist):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=HIST_FIELDS)
        w.writeheader()
        for r in sorted(hist.values(), key=lambda r: (r["start"], r["id"])):
            w.writerow({k: r.get(k, "") for k in HIST_FIELDS})

def log_predictions(hist, matches, now):
    """Upsert each not-yet-started match, so the log keeps the last price seen before the start."""
    # A match first priced from api-tennis and later by the odds feed has two ids;
    # keep one row, the odds feed's.
    def same(h):
        return (h["tour"], (h.get("start") or "")[:10], frozenset((loose_key(h["player_a"]), loose_key(h["player_b"]))))
    feed = {same({"tour": m["tour"], "start": m["start"], "player_a": m["a"]["name"], "player_b": m["b"]["name"]})
            for m in matches if not m["id"].startswith("apit-")}
    for k in [k for k, h in hist.items() if k.startswith("apit-") and not h.get("result") and same(h) in feed]:
        del hist[k]
    for m in matches:
        st = parse_iso(m["start"])
        if not st or st <= now:
            continue
        old = hist.get(m["id"], {})
        if old.get("result"):
            continue
        hist[m["id"]] = {"id": m["id"], "tour": m["tour"], "tournament": m["tournament"], "start": m["start"],
                         "surface": m["surface"], "best_of": m["best_of"],
                         "player_a": m["a"]["name"], "player_b": m["b"]["name"],
                         "model_a": m["a"]["model"], "market_a": m["a"]["market"],
                         "odds_a": m["a"]["best_odds"], "odds_b": m["b"]["best_odds"],
                         "books": m.get("books", ""), "logged": iso(now), "result": "", "settled": "",
                         "pinnacle_a": m.get("pinnacle_a", ""),
                         "flags": ";".join(sorted({c for c, _ in m.get("flags", [])})),
                         "model_version": m.get("model_version", "")}

def settle(hist, results, now):
    """Fill in winners from recent_results.json. result: 'a', 'b' or 'void'."""
    by_pair = {}
    for r in results:
        key = (r["tour"], frozenset((loose_key(r["winner"]), loose_key(r["loser"]))))
        by_pair.setdefault(key, []).append(r)
    n = 0
    for h in hist.values():
        if h.get("result"):
            continue
        st = parse_iso(h["start"])
        if not st or now - st < timedelta(hours=SETTLE_AFTER_H):
            continue
        ka, kb = name_keys(h["player_a"]), name_keys(h["player_b"])
        cands = [r for x in ka for y in kb for r in by_pair.get((h["tour"], frozenset((x, y))), [])]
        # Sackmann dates each match by the tournament's start, so allow two weeks before the start time.
        lo, hi = (st - timedelta(days=15)).date().isoformat(), (st + timedelta(days=3)).date().isoformat()
        cands = [r for r in cands if lo <= r["date"] <= hi]
        if cands:
            r = cands[-1]
            h["result"] = "void" if r["status"] == "wo" else ("a" if loose_key(r["winner"]) in ka else "b")
            h["settled"] = iso(now)
            n += 1
        elif now - st > timedelta(days=GIVE_UP_DAYS):
            h["result"], h["settled"] = "void", iso(now)
    return n

def track_record(hist, split=True):
    rows = [h for h in hist.values() if h.get("result") in ("a", "b")]
    rec = {"settled": len(rows), "pending": sum(1 for h in hist.values() if not h.get("result"))}
    if not rows:
        return rec
    if split:
        # the same numbers per model version, so a model change gets its own record
        versions = sorted({h.get("model_version") or "original" for h in rows})
        rec["by_model"] = {v: track_record({k: h for k, h in hist.items()
                                            if (h.get("model_version") or "original") == v}, split=False)
                           for v in versions}
    f = lambda h, k: float(h[k])
    won_a = lambda h: 1.0 if h["result"] == "a" else 0.0
    rec["model_right"] = round(sum((f(h, "model_a") > 0.5) == (h["result"] == "a") for h in rows) / len(rows), 4)
    rec["market_right"] = round(sum((f(h, "market_a") > 0.5) == (h["result"] == "a") for h in rows) / len(rows), 4)
    rec["brier_model"] = round(statistics.mean((f(h, "model_a") - won_a(h)) ** 2 for h in rows), 4)
    rec["brier_market"] = round(statistics.mean((f(h, "market_a") - won_a(h)) ** 2 for h in rows), 4)
    # flat 1-unit bet on the better +EV side of each match, at the logged best price
    bets = profit = wins = 0
    for h in rows:
        pa, oa, ob = f(h, "model_a"), f(h, "odds_a"), f(h, "odds_b")
        ev_a, ev_b = pa * oa - 1, (1 - pa) * ob - 1
        if max(ev_a, ev_b) <= 0:
            continue
        pick_a = ev_a >= ev_b
        bets += 1
        if pick_a == (h["result"] == "a"):
            wins += 1; profit += (oa if pick_a else ob) - 1
        else:
            profit -= 1
    rec.update(value_bets=bets, value_wins=wins, profit_units=round(profit, 2),
               roi=round(profit / bets, 4) if bets else None)
    # against Pinnacle, on the matches Pinnacle priced
    pin = [h for h in rows if h.get("pinnacle_a")]
    if pin:
        rec["pinnacle_n"] = len(pin)
        rec["brier_model_pin"] = round(statistics.mean((f(h, "model_a") - won_a(h)) ** 2 for h in pin), 4)
        rec["brier_pinnacle"] = round(statistics.mean((f(h, "pinnacle_a") - won_a(h)) ** 2 for h in pin), 4)
    # value bets on matches with no warnings
    # (only predictions logged since warnings existed: they carry a model version)
    clean = [h for h in rows if h.get("model_version") and not h.get("flags")]
    cb = cp = 0
    for h in clean:
        pa, oa, ob = f(h, "model_a"), f(h, "odds_a"), f(h, "odds_b")
        ev_a, ev_b = pa * oa - 1, (1 - pa) * ob - 1
        if max(ev_a, ev_b) <= 0:
            continue
        pick_a = ev_a >= ev_b
        cb += 1
        cp += ((oa if pick_a else ob) - 1) if pick_a == (h["result"] == "a") else -1
    rec.update(clean_bets=cb, clean_profit=round(cp, 2))
    return rec


# ---------- Kalshi (prediction market) ----------------------------------------
# Kalshi lists match-winner contracts: one market per player, paying $1 if they
# win, priced 1-99 cents. Market data is public, so no key is needed. Kalshi takes
# a trading fee of about 7% x p x (1 - p) per contract, which is folded into the
# odds shown so EV is what you'd actually get.
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_FEE = 0.07
KALSHI_SERIES_FALLBACK = ("KXATPMATCH", "KXWTAMATCH", "KXATPCHALLENGERMATCH", "KXWTACHALLENGERMATCH")

def kalshi_get(path, **params):
    q = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    req = urllib.request.Request(f"{KALSHI}{path}?{q}", headers={"Accept": "application/json",
                                                                 "User-Agent": "tennis-odds-engine"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)

def kalshi_price(m, side):
    """Ask price in dollars for 'yes' or 'no' (newer replies use *_dollars strings)."""
    v = m.get(f"{side}_ask_dollars")
    if v not in (None, ""):
        try:
            return float(v)
        except ValueError:
            pass
    v = m.get(f"{side}_ask")
    return v / 100 if isinstance(v, (int, float)) else None

def kalshi_mid(m):
    bid, ask = m.get("yes_bid_dollars") or m.get("yes_bid"), kalshi_price(m, "yes")
    try:
        bid = float(bid) if isinstance(bid, str) else (bid / 100 if bid is not None else None)
    except ValueError:
        bid = None
    if ask and bid:
        return (ask + bid) / 2
    return ask

def kalshi_odds(cost):
    """Decimal odds after Kalshi's fee for a contract bought at `cost` dollars."""
    if not cost or not (0 < cost < 1):
        return None
    return round(1 / (cost + KALSHI_FEE * cost * (1 - cost)), 3)

def fetch_kalshi(now, hours):
    """[{tour, event, series, url, close, players: [(name, cost, mid, ticker), ...]}] for open
    two-player tennis match events; [] if Kalshi can't be reached."""
    try:
        series = kalshi_get("/series", category="Sports").get("series") or []
    except Exception as e:                                        # noqa: BLE001
        print(f"  · Kalshi: unavailable ({e})", file=sys.stderr)
        return []
    tennis = [x for x in series if "tennis" in json.dumps(x).lower()]
    match = [x for x in tennis if re.search(r"match|vs|winner", (x.get("title", "") + x.get("ticker", "")), re.I)
             and not re.search(r"tournament|champion|title|outright|set|game|total", x.get("title", ""), re.I)]
    if not match:
        match = [{"ticker": t, "title": t} for t in KALSHI_SERIES_FALLBACK]
    print(f"  · Kalshi: {len(tennis)} tennis series, using {[x.get('ticker') for x in match][:8]}", file=sys.stderr)
    end = now + timedelta(hours=hours + 24)
    out = []
    for sr in match:
        ticker, title = sr.get("ticker", ""), sr.get("title", "")
        tour = "WTA" if re.search(r"wta|women", ticker + " " + title, re.I) else "ATP"
        events, cursor = {}, None
        for _ in range(10):                                       # pages of up to 1000 markets
            try:
                page = kalshi_get("/markets", series_ticker=ticker, status="open", limit=1000, cursor=cursor)
            except Exception as e:                                # noqa: BLE001
                print(f"  · Kalshi {ticker}: {e}", file=sys.stderr)
                break
            for m in page.get("markets") or []:
                events.setdefault(m.get("event_ticker"), []).append(m)
            cursor = page.get("cursor")
            if not cursor:
                break
        for ev, ms in events.items():
            if len(ms) != 2:
                continue                                          # not a plain head-to-head
            close = parse_iso(ms[0].get("expected_expiration_time") or ms[0].get("close_time") or "")
            if close and close > end + timedelta(days=7):
                continue
            players = []
            for m in ms:
                name = (m.get("yes_sub_title") or "").strip()
                players.append((name, kalshi_price(m, "yes"), kalshi_mid(m), m.get("ticker")))
            if not all(p[0] for p in players):
                continue
            slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or ticker.lower()
            out.append({"tour": tour, "event": ev, "series": ticker, "close": close, "players": players,
                        "url": f"https://kalshi.com/markets/{ticker.lower()}/{slug}/{(ev or '').lower()}"})
    print(f"  · Kalshi: {len(out)} open head-to-head match events", file=sys.stderr)
    return out

def attach_kalshi(matches, kal, players, idx):
    """Add Kalshi prices to the matches they belong to (matched on the two players)."""
    by_pair = {}
    for m in matches:
        by_pair[(m["tour"], frozenset((m["a"].get("key"), m["b"].get("key"))))] = m
    n = 0
    for k in kal:
        (na, ca, ma, ta), (nb, cb, mb, tb) = k["players"]
        ka, kb = resolve(na, k["tour"], players, idx), resolve(nb, k["tour"], players, idx)
        m = by_pair.get((k["tour"], frozenset((ka, kb))))
        if not m or not (ka and kb):
            continue
        st = parse_iso(m.get("start"))
        if st and k["close"] and abs((k["close"] - st).total_seconds()) > 4 * 86400:
            continue                                              # same players, a different match
        if ka != m["a"].get("key"):
            (na, ca, ma, ta), (nb, cb, mb, tb) = (nb, cb, mb, tb), (na, ca, ma, ta)
        mkt_a = ma / (ma + mb) if ma and mb else None
        sides = {}
        for sd, cost, model_p in (("a", ca, m["a"]["model"]), ("b", cb, m["b"]["model"])):
            odds = kalshi_odds(cost)
            sides[sd] = {"cost": cost, "odds": odds,
                         "ev": round(model_p * odds - 1, 4) if odds else None,
                         "ticker": ta if sd == "a" else tb}
        m["kalshi"] = {**sides, "market_a": round(mkt_a, 4) if mkt_a else None,
                       "event": k["event"], "url": k["url"]}
        n += 1
    return n

def kalshi_parlay_matches(matches):
    """The matches as Kalshi-only markets, so build_parlays can price tickets at Kalshi."""
    out = []
    for m in matches:
        k = m.get("kalshi")
        if not k or not (k["a"]["odds"] and k["b"]["odds"]) or not k.get("market_a"):
            continue
        mm = json.loads(json.dumps(m))
        mm["prices"] = {"Kalshi": [k["a"]["odds"], k["b"]["odds"]]}
        mm["a"]["market"], mm["b"]["market"] = k["market_a"], 1 - k["market_a"]
        out.append(mm)
    return out


# ---------- schedule (api-tennis.com) ----------------------------------------
# The odds feed only lists matches a bookmaker has priced, so the order of play
# for tomorrow is often half missing there. api-tennis has the full schedule;
# matches it lists that the odds feed doesn't get the model's number, no odds.
def fetch_schedule(key, now, hours):
    """Upcoming ATP/WTA singles from api-tennis: [{tour, tournament, start, a, b, ...}]."""
    from update_data import api_tennis_call, API_TENNIS_TYPES      # lazy: update_data imports this module
    end = now + timedelta(hours=hours)
    out = []
    for tour, (type_key, type_name) in API_TENNIS_TYPES.items():
        params = dict(method="get_fixtures", event_type_key=type_key,
                      date_start=(now - timedelta(days=1)).date().isoformat(),
                      date_stop=(end + timedelta(days=1)).date().isoformat())
        try:
            events, tz = api_tennis_call(key, timezone="Etc/UTC", **params), timezone.utc
        except Exception as e:                                        # noqa: BLE001
            print(f"  · schedule {tour.upper()}: {e}", file=sys.stderr)
            continue
        for ev in events:
            if ev.get("event_type_type") != type_name or ev.get("event_live") == "1":
                continue
            if (ev.get("event_status") or "").strip() not in ("", "Not Started"):
                continue                                              # finished, retired, cancelled
            a, b = (ev.get("event_first_player") or "").strip(), (ev.get("event_second_player") or "").strip()
            if not a or not b:
                continue
            try:
                start = datetime.strptime(f"{ev.get('event_date')} {ev.get('event_time') or '00:00'}",
                                          "%Y-%m-%d %H:%M").replace(tzinfo=tz)
            except ValueError:
                continue
            if not (now - timedelta(hours=14) < start <= end + timedelta(hours=14)):
                continue                       # final window check after any time-zone fix
            out.append({"key": str(ev.get("event_key")), "tour": tour.upper(),
                        "tournament": ev.get("tournament_name", ""), "round": ev.get("tournament_round", ""),
                        "qualifying": ev.get("event_qualification") == "True", "start": start,
                        "a": a, "a_key": ev.get("first_player_key"), "b": b, "b_key": ev.get("second_player_key")})
    return out

# api-tennis also sells bookmaker odds, covering far more matches (qualifying,
# smaller events) than the odds feed. Used only for matches the odds feed lacks.
WIN_MARKETS = ("home/away", "match winner", "to win match", "winner", "1x2")

def fetch_api_tennis_odds(key, now, hours):
    """{event_key: {bookmaker: [price first player, price second player]}}; {} if unavailable."""
    from update_data import api_tennis_call
    end = now + timedelta(hours=hours)
    try:
        res = api_tennis_call(key, method="get_odds", date_start=now.date().isoformat(),
                              date_stop=end.date().isoformat())
    except Exception as e:                                        # noqa: BLE001
        print(f"  · api-tennis odds: not available ({e})", file=sys.stderr)
        return {}
    if not isinstance(res, dict):
        print(f"  · api-tennis odds: unexpected reply ({type(res).__name__})", file=sys.stderr)
        return {}
    out, markets = {}, {}
    for ev_key, mk in res.items():
        if not isinstance(mk, dict):
            continue
        for name in mk:
            markets[name] = markets.get(name, 0) + 1
        win = next((mk[n] for n in mk if n.strip().lower() in WIN_MARKETS), None)
        if not isinstance(win, dict):
            continue
        sides = {k.strip().lower(): v for k, v in win.items() if isinstance(v, dict)}
        home, away = sides.get("home") or sides.get("1"), sides.get("away") or sides.get("2")
        if not (home and away):
            continue
        books = {}
        for bk in home:
            try:
                pa, pb = float(home[bk]), float(away.get(bk))
            except (TypeError, ValueError):
                continue
            if pa > 1 and pb > 1:
                books[bk] = [pa, pb]
        if books:
            out[str(ev_key)] = books
    top = ", ".join(f"{n} ({c})" for n, c in sorted(markets.items(), key=lambda x: -x[1])[:6])
    print(f"  · api-tennis odds: {len(res)} events, {len(out)} with match-winner prices; markets: {top or 'none'}",
          file=sys.stderr)
    return out

def summary_from_books(books):
    """The odds-feed summary (best prices, de-vigged market, books, Pinnacle, parlay books)
    from {bookmaker: [price A, price B]}."""
    best_a, best_b, fair, pin, by_book = (0.0, None), (0.0, None), [], None, {}
    for bk, (pa, pb) in books.items():
        if pa > best_a[0]: best_a = (pa, bk)
        if pb > best_b[0]: best_b = (pb, bk)
        ia, ib = 1 / pa, 1 / pb
        fair.append(ia / (ia + ib))
        low = bk.lower()
        if not any(x in low for x in ("exchange", "betfair", "matchbook", "smarkets", "betdaq")):
            by_book[bk] = [pa, pb]
        if "pinnacle" in low or low == "pncl":      # api-tennis abbreviates it "Pncl"
            pin = ia / (ia + ib)
    return best_a, best_b, statistics.mean(fair), len(fair), pin, by_book

def schedule_pid(name, api_key, players):
    """players.json id for an api-tennis name (the stats come from the same feed)."""
    for pid in (slugify(f"{name} ({api_key})"), slugify(name)):
        if pid in players:
            return pid
    return None

PLACEHOLDER = re.compile(r"\b(tba|tbd|bye|qualifier|lucky loser|winner)\b|/", re.I)

def newcomer(name, tour):
    """A player with no matches in the stats yet (often in qualifying): the rating a
    first-time qualifier starts on, tour-average serve and return, flagged thin."""
    try:
        with open("model_params.json", encoding="utf-8") as fh:
            elo = json.load(fh).get("params", {}).get("init_elo_qual", 1250)
    except (OSError, ValueError):
        elo = 1250
    return {"name": name, "tour": tour, "elo": {"overall": elo}, "n": 0, "new": True,
            "info": {"m12": 0, "surf12": {}}}

def add_schedule(out, sched, players, tour_avgs, mdl, version, today, hours=36, odds=None):
    """Add the scheduled matches the odds feed doesn't have, with the model's number only."""
    have = {frozenset((m["a"].get("key"), m["b"].get("key"))): m for m in out["matches"]}
    # If api-tennis ignored the time zone asked for, its times are off by whole hours;
    # the matches both feeds list show by how much.
    diffs = sorted(round((f["start"] - parse_iso(have[pair]["start"])).total_seconds() / 3600)
                   for f in sched
                   for pair in [frozenset((schedule_pid(f["a"], f["a_key"], players), schedule_pid(f["b"], f["b_key"], players)))]
                   if pair in have and parse_iso(have[pair]["start"]))
    # (newcomers can't be in the odds list under a players.json id, so they don't count here)
    shift = diffs[len(diffs) // 2] if len(diffs) >= 3 and abs(diffs[len(diffs) // 2]) <= 14 else 0
    if shift:
        print(f"  · schedule times are {shift:+d}h from the odds feed's ({len(diffs)} matches in both); corrected",
              file=sys.stderr)
        sched = [{**f, "start": f["start"] - timedelta(hours=shift)} for f in sched]
    now = datetime.now(timezone.utc)
    unmatched = [(name_keys(u["a"]["name"]), name_keys(u["b"]["name"])) for u in out["unmatched"]]
    added = 0
    extra = {}                             # newcomers, by id, for this run only
    for f in sched:
        if PLACEHOLDER.search(f["a"]) or PLACEHOLDER.search(f["b"]):
            continue                       # draw slot not filled yet
        ids = []
        for name, akey in ((f["a"], f["a_key"]), (f["b"], f["b_key"])):
            pid = schedule_pid(name, akey, players)
            if not pid:
                pid = "new-" + slugify(f"{name} {akey}")
                extra.setdefault(pid, newcomer(name, f["tour"]))
            ids.append(pid)
        ka, kb = ids
        if frozenset((ka, kb)) in have or not (now < f["start"] <= now + timedelta(hours=hours)):
            continue
        fa, fb = name_keys(f["a"]), name_keys(f["b"])
        if any((ua & fa and ub & fb) or (ua & fb and ub & fa) for ua, ub in unmatched):
            continue                       # priced, but the odds feed's names didn't resolve
        month = f["start"].month
        surf, indoor, known = surface_for(f["tournament"], f["tour"], month)
        best_of = 3 if f["qualifying"] else guess_best_of(f["tournament"], f["tour"])
        t_avg = tour_avgs.get(f["tour"]) or (0.64 if f["tour"] == "ATP" else 0.56)
        A, B = players.get(ka) or extra[ka], players.get(kb) or extra[kb]
        r = predict_export(A, B, surf, best_of, t_avg, mdl)
        p = r["p"]
        def no_odds(name, key, q):
            return {"name": name, "key": key, "model": round(q, 4), "fair_odds": round(1 / q, 2),
                    "market": None, "best_odds": None, "book": None, "ev": None, "kelly": None}
        title = f"{f['tour']} {f['tournament']}" + (" (qualifying)" if f["qualifying"] else "")
        row = {"id": "sched-" + f["key"], "tour": f["tour"], "tournament": title, "round": f["round"],
               "start": iso(f["start"]), "surface": surf, "indoor": indoor, "surface_known": known,
               "best_of": best_of, "a": no_odds(A["name"], ka, p),
               "b": no_odds(B["name"], kb, 1 - p), "books": 0, "no_odds": True,
               "court": court_id(f["tournament"], f["tour"], month), "model_version": version,
               "parts": {"elo": round(r["p_elo"], 4), "serve": round(r["p_mkv"], 4), "w": round(r["w"], 3)},
               "flags": [fl for fl in match_flags(A, B, surf, today, p, p)]}
        books = (odds or {}).get(f["key"])
        if books:                          # priced by api-tennis's bookmakers
            best_a, best_b, mkt_a, n_books, pin_a, by_book = summary_from_books(books)
            row["a"].update(side(p, mkt_a, *best_a)); row["b"].update(side(1 - p, 1 - mkt_a, *best_b))
            row.update(books=n_books, prices=by_book, no_odds=False, odds_source="api-tennis",
                       id="apit-" + f["key"],
                       flags=match_flags(A, B, surf, today, p, pin_a if pin_a is not None else mkt_a))
            if pin_a is not None:
                row["pinnacle_a"] = round(pin_a, 4)
        # the site needs a newcomer's numbers to load the match into the predictor
        new = {s_: P for s_, P in (("a", A), ("b", B)) if P.get("new")}
        if new:
            row["new_players"] = new
            for s_, P in new.items():
                row[s_]["new"] = True
        out["matches"].append(row)
        have[frozenset((ka, kb))] = row
        added += 1
    out["matches"].sort(key=lambda m: m["start"] or "")
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", default="players.json")
    ap.add_argument("--out", default="matches.json")
    ap.add_argument("--results", default="recent_results.json")
    ap.add_argument("--history", default="history/predictions.csv")
    ap.add_argument("--parlay-history", default="history/parlays.csv")
    ap.add_argument("--hours", type=int, default=36, help="look this far ahead")
    ap.add_argument("--regions", default="eu", help="bookmaker regions: us, uk, eu, au (each costs 1 request)")
    ap.add_argument("--min-remaining", type=int, default=0,
                    help="skip fetching odds if the monthly allowance left is below this")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    out = {"updated": iso(now), "window_hours": args.hours, "status": "ok",
           "requests_remaining": None, "matches": [], "unmatched": []}
    hist = load_history(args.history)
    try:
        with open(args.results, encoding="utf-8") as fh:
            results = json.load(fh).get("results", [])
    except (OSError, ValueError):
        results = []

    parlay_rows = load_parlays(args.parlay_history)

    def finish():
        tkey = os.environ.get("TENNIS_API_KEY", "").strip()
        if tkey and players:
            sched = fetch_schedule(tkey, now, args.hours)
            odds = fetch_api_tennis_odds(tkey, now, args.hours)
            n = add_schedule(out, sched, players, tour_avgs, mdl, version, now.date(), args.hours, odds)
            n_priced = sum(1 for m in out["matches"] if m.get("odds_source") == "api-tennis")
            out["scheduled_only"] = n - n_priced
            n_new = sum(1 for m in out["matches"] if m.get("new_players"))
            print(f"Schedule: {len(sched)} upcoming on api-tennis, {n} added: {n_priced} priced by "
                  f"api-tennis, {n - n_priced} without odds ({n_new} with a player new to our data)",
                  file=sys.stderr)
        if players:
            kal = fetch_kalshi(now, args.hours)
            out["kalshi_matched"] = attach_kalshi(out["matches"], kal, players, idx)
            print(f"Kalshi: {out['kalshi_matched']} of {len(kal)} Kalshi matches paired with the list", file=sys.stderr)
        priced = [m for m in out["matches"] if not m.get("no_odds")]
        n_settled = settle(hist, results, now)
        log_predictions(hist, priced, now)
        out["record"] = track_record(hist)
        save_history(args.history, hist)
        # api-tennis-priced matches can change id when the odds feed picks them up, which
        # would orphan a logged parlay leg, so recommended parlays use the odds feed only
        fresh = build_parlays([m for m in priced if m.get("odds_source") != "api-tennis"], now)
        settle_parlays(parlay_rows, hist, now)
        # Three a day: once the day's three are logged, later runs show those same
        # tickets (at their logged prices) instead of a new set from the latest odds.
        def logged_today(kal):
            return sum(1 for r in parlay_rows.values()
                       if r["created"][:10] == now.date().isoformat() and is_kalshi(r) == kal)
        log_parlays(parlay_rows, [c for c in fresh if c["id"] not in parlay_rows][:max(0, PARLAY_SHOW - logged_today(False))], now)
        out["parlays"] = todays_parlays(parlay_rows, hist, out["matches"], now)
        # Kalshi gets its own three a day, priced at Kalshi (after its fee)
        # (only matches with a lasting id, so a logged leg can always be settled)
        kfresh = build_parlays(kalshi_parlay_matches([m for m in priced if m.get("odds_source") != "api-tennis"]), now)
        for c in kfresh:
            c["id"] = "kalshi|" + "+".join(f"{l['id']}:{l['side']}" for l in c["legs"])
        log_parlays(parlay_rows, [c for c in kfresh if c["id"] not in parlay_rows][:max(0, PARLAY_SHOW - logged_today(True))], now)
        out["kalshi_parlays"] = todays_parlays(parlay_rows, hist, out["matches"], now, kalshi=True)
        out["parlay_record"] = parlay_record(parlay_rows)
        save_parlays(args.parlay_history, parlay_rows)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
        print(f"History: {len(hist)} logged, {n_settled} newly settled; record {out['record']}", file=sys.stderr)

    try:
        with open(args.players, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        data = {"players": {}}
    players = data["players"]
    idx = build_name_index(players)
    tour_avgs = {t.upper(): v.get("tour_avg_spw") for t, v in (data.get("tours") or {}).items()}
    mdl = data.get("model") or {"rating": "elo", "w": {"ATP": 0.5, "WTA": 0.5}, "n0": 0}
    version = mdl.get("version", "")
    today = now.date()

    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        out["status"] = "no_api_key"
        print("ODDS_API_KEY not set — no odds fetched.", file=sys.stderr)
        return finish()

    sports, remaining = get_json("/sports", {"apiKey": key})       # this call is free
    tennis = [s for s in sports if s.get("active") and re.match(r"tennis_(atp|wta)_", s.get("key", ""))]
    print(f"{len(tennis)} active ATP/WTA events, requests remaining: {remaining}", file=sys.stderr)
    if remaining is not None and args.min_remaining and int(float(remaining)) < args.min_remaining:
        # Keep the morning's list rather than spend the last of the allowance.
        try:
            with open(args.out, encoding="utf-8") as fh:
                prev = json.load(fh)
            out["matches"] = [m for m in prev.get("matches", []) if not m.get("no_odds")]
            out["unmatched"], out["updated"] = prev.get("unmatched", []), prev.get("updated")
        except (OSError, ValueError):
            pass
        out["status"], out["requests_remaining"] = "quota_low", remaining
        print(f"Only {remaining} requests left this month — skipped the odds refresh.", file=sys.stderr)
        return finish()

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

        best_of = guess_best_of(title, tour)
        t_avg = tour_avgs.get(tour) or (0.64 if tour == "ATP" else 0.56)
        for ev in events:
            start = parse_iso(ev.get("commence_time"))
            month = start.month if start else now.month
            surf, indoor, known = surface_for(title, tour, month)
            court = court_id(title, tour, month)
            if not known:
                print(f"  · {title}: surface unknown, assuming hard", file=sys.stderr)
            a, b = ev.get("home_team"), ev.get("away_team")
            ka, kb = resolve(a, tour, players, idx), resolve(b, tour, players, idx)
            summary = price_summary(ev, a, b)
            row = {"id": ev.get("id"), "tour": tour, "tournament": title,
                   "start": ev.get("commence_time"), "surface": surf, "indoor": indoor,
                   "surface_known": known, "best_of": best_of,
                   "a": {"name": a, "key": ka}, "b": {"name": b, "key": kb}}
            if not summary:
                continue
            (best, mkt_a, n_books, pin_a, by_book) = summary
            row["books"] = n_books
            row["prices"] = by_book
            if not (ka and kb):
                row["missing"] = [n for n, k in ((a, ka), (b, kb)) if not k]
                out["unmatched"].append(row)
                continue
            r = predict_export(players[ka], players[kb], surf, best_of, t_avg, mdl)
            p = r["p"]
            row["a"].update(side(p, mkt_a, *best[a]))
            row["b"].update(side(1 - p, 1 - mkt_a, *best[b]))
            row.update(court=court, model_version=version,
                       parts={"elo": round(r["p_elo"], 4), "serve": round(r["p_mkv"], 4), "w": round(r["w"], 3)})
            if pin_a is not None:
                row["pinnacle_a"] = round(pin_a, 4)
            row["flags"] = match_flags(players[ka], players[kb], surf, today, p,
                                       pin_a if pin_a is not None else mkt_a)
            out["matches"].append(row)

    out["matches"].sort(key=lambda r: r["start"] or "")
    out["unmatched"].sort(key=lambda r: r["start"] or "")
    out["requests_remaining"] = remaining
    n_val = sum(1 for m in out["matches"] if m["a"]["ev"] > 0 or m["b"]["ev"] > 0)
    print(f"{len(out['matches'])} matches ({n_val} with a +EV side), "
          f"{len(out['unmatched'])} unmatched, requests remaining: {remaining}", file=sys.stderr)
    finish()


if __name__ == "__main__":
    main()
