"""Daily data updater for the Tennis Odds Engine (ATP + WTA).

Pulls recent ATP and WTA match results, computes current surface ELO and
serve/return rates per player, and writes players.json — the file the web app
reads on load. Each player is tagged with their tour, and each tour carries its
own average serve-points-won baseline (WTA serve hold rates run lower than ATP),
so the Markov math uses the right reference for each.

Sources, tried in order for each tour:
  1. api-tennis.com (needs TENNIS_API_KEY) — results plus serve points won.
     Matches are cached in data/api_tennis_matches.csv: the first run
     backfills about two years, later runs fetch only the last week, and the
     cache keeps the history even if the API stops responding.
  2. Jeff Sackmann's match CSVs on GitHub — results plus serve/return stats.
  3. tennis-data.co.uk yearly spreadsheets — results only. ELO stays current;
     serve/return rates are carried over from the previous players.json
     (new players get the tour average) and flagged as carried over.
  4. None reachable — the previous players.json is kept as-is and marked
     stale, so the site keeps working and says the data is old.

Also writes recent_results.json (the last few weeks of results), which
update_matches.py uses to settle its logged predictions.

Model details:
  - ELO ignores walkovers and retirements (they say little about strength).
  - Serve/return rates are points-weighted, recency-weighted (half-life one
    year) and adjusted for opponent strength.
  - A player is listed if they played in the last 12 months.

Self-contained (standard library only) so it runs cleanly on a CI runner.

Usage:
    python update_data.py                      # ATP + WTA, last 3 seasons
    python update_data.py --tours atp          # one tour only
    python update_data.py --years 2024 2025 2026 --top 60
    python update_data.py --local sample.csv --tours wta   # test a Sackmann-format CSV
    python update_data.py --local-xlsx 2025.xlsx --tours atp  # test a tennis-data file
"""

import argparse, csv, io, json, os, re, sys, unicodedata, zipfile
import urllib.parse, urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone

SACKMANN = {
    "atp": {"repo": "JeffSackmann/tennis_atp", "file": "atp_matches_{year}.csv"},
    "wta": {"repo": "JeffSackmann/tennis_wta", "file": "wta_matches_{year}.csv"},
}
RAW = "https://raw.githubusercontent.com/{repo}/{branch}/{file}"
TENNIS_DATA = {
    "atp": "http://www.tennis-data.co.uk/{year}/{year}.xlsx",
    "wta": "http://www.tennis-data.co.uk/{year}w/{year}.xlsx",
}
UA = {"User-Agent": "tennis-odds-engine/1.0 (+github actions daily refresh)"}

SURFACES = ("hard", "clay", "grass")
DEFAULT_ELO = 1500.0
MIN_SURFACE_MATCHES = 3
RECENT_DAYS = 365          # listed if played within this many days of the latest result
HALF_LIFE_DAYS = 365       # recency weighting for serve/return rates
SHRINK_POINTS = 400        # opponent ratings shrink toward the tour average by this many points
RESULTS_DAYS = 28          # how much history recent_results.json keeps
STALE_AFTER_DAYS = 14      # newest result older than this -> tour marked stale


# ---------- tiny ELO ----------------------------------------------------------
def k_factor(m): return 250.0 / (m + 5) ** 0.4
def expected(ra, rb): return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


# ---------- helpers -----------------------------------------------------------
def slugify(name):
    n = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", n.lower()).strip("-")

def loose_key(name):
    """'Jannik Sinner' / 'J. Sinner' -> 'j-sinner'. Used to match across sources."""
    parts = slugify(name).split("-")
    return parts[0][:1] + "-" + "-".join(parts[1:]) if len(parts) >= 2 else slugify(name)

def norm_surface(s):
    s = (s or "").strip().lower()
    if s in SURFACES:
        return s
    return "hard" if s == "carpet" else None

def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def parse_ymd(s):
    try:
        return datetime.strptime(str(s)[:8], "%Y%m%d").date()
    except ValueError:
        return None


def http_get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as resp:
        return resp.read()


# ---------- source 1: api-tennis.com ------------------------------------------
API_TENNIS = "https://api.api-tennis.com/tennis/"
API_TENNIS_TYPES = {"atp": ("265", "Atp Singles"), "wta": ("266", "Wta Singles")}
API_CACHE_FIELDS = ["event_key", "date", "time", "tour", "tournament", "round", "qualifying",
                    "surface", "indoor", "status", "winner", "winner_key", "loser", "loser_key",
                    "w_srv_won", "w_srv_pts", "l_srv_won", "l_srv_pts"]
API_STATUS = {"Finished": "completed", "Retired": "ret", "Walk Over": "wo"}
API_WINDOW_DAYS = 7        # date range per request
API_REFRESH_DAYS = 7       # re-fetch this many recent days each run (late stats, corrections)
API_BACKFILL_DAYS = 730    # history fetched on the first run

def api_tennis_call(key, **params):
    q = urllib.parse.urlencode({"APIkey": key, **params})
    data = json.loads(http_get(f"{API_TENNIS}?{q}").decode("utf-8", "replace"))
    if not data.get("success"):
        raise RuntimeError(str(data.get("error") or data)[:300].replace(key, "***"))
    return data.get("result") or []

def _service_points(ev, player_key):
    for st in ev.get("statistics") or []:
        if (st.get("player_key") == player_key and st.get("stat_period") == "match"
                and st.get("stat_name") == "Service Points Won"):
            won, tot = to_float(st.get("stat_won")), to_float(st.get("stat_total"))
            if won is not None and tot:
                return int(won), int(tot)
    return None

def api_event_row(ev, tour):
    """One api-tennis fixture -> cache row, or None if it isn't a played singles match."""
    from update_matches import surface_for       # same tournament table as the match list
    status = API_STATUS.get(ev.get("event_status"))
    winner_side = ev.get("event_winner")
    if not status or winner_side not in ("First Player", "Second Player"):
        return None
    first = (ev.get("event_first_player"), ev.get("first_player_key"))
    second = (ev.get("event_second_player"), ev.get("second_player_key"))
    (w, wk), (l, lk) = (first, second) if winner_side == "First Player" else (second, first)
    if not w or not l:
        return None
    surf, indoor, _ = surface_for(ev.get("tournament_name", ""), tour.upper())
    ws, ls = _service_points(ev, wk), _service_points(ev, lk)
    return {"event_key": ev.get("event_key"), "date": ev.get("event_date"), "time": ev.get("event_time") or "",
            "tour": tour.upper(), "tournament": ev.get("tournament_name", ""),
            "round": ev.get("tournament_round", ""), "qualifying": ev.get("event_qualification") == "True",
            "surface": surf, "indoor": indoor, "status": status,
            "winner": w.strip(), "winner_key": wk, "loser": l.strip(), "loser_key": lk,
            "w_srv_won": ws[0] if ws else "", "w_srv_pts": ws[1] if ws else "",
            "l_srv_won": ls[0] if ls else "", "l_srv_pts": ls[1] if ls else ""}

def load_api_cache(path):
    try:
        with open(path, encoding="utf-8", newline="") as fh:
            return {r["event_key"]: r for r in csv.DictReader(fh)}
    except OSError:
        return {}

def save_api_cache(path, cache):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=API_CACHE_FIELDS)
        w.writeheader()
        for r in sorted(cache.values(), key=lambda r: (r["date"], r["time"], str(r["event_key"]))):
            w.writerow({k: r.get(k, "") for k in API_CACHE_FIELDS})

def sync_api_tennis(key, cache_path, tours, today=None):
    """Bring the cache up to date. Returns (cache, number of new/updated rows, error or None)."""
    today = today or date.today()
    cache = load_api_cache(cache_path)
    changed, err = 0, None
    for tour in tours:
        type_key, type_name = API_TENNIS_TYPES[tour]
        # Per tour, so an interrupted backfill of one tour resumes where it stopped.
        have = [r["date"] for r in cache.values() if r["tour"] == tour.upper()]
        a = (datetime.strptime(max(have), "%Y-%m-%d").date() - timedelta(days=API_REFRESH_DAYS)
             if have else today - timedelta(days=API_BACKFILL_DAYS))
        while a <= today:
            b = min(a + timedelta(days=API_WINDOW_DAYS - 1), today)
            try:
                events = api_tennis_call(key, method="get_fixtures", event_type_key=type_key,
                                         date_start=a.isoformat(), date_stop=b.isoformat())
            except Exception as e:                   # noqa: BLE001
                err = f"{tour} {a}..{b}: {e}"
                print(f"  · api-tennis {err}", file=sys.stderr)
                break                                # keep what we have; retry tomorrow
            for ev in events:
                if ev.get("event_type_type") != type_name:
                    continue
                row = api_event_row(ev, tour)
                if row:
                    k = str(row["event_key"])
                    row = {f: ("" if row[f] is None else str(row[f])) for f in API_CACHE_FIELDS}
                    if cache.get(k) != row:
                        cache[k] = row
                        changed += 1
            a = b + timedelta(days=1)
    save_api_cache(cache_path, cache)
    return cache, changed, err

def from_api_cache(cache, tour_label):
    out, names = [], {}
    for r in cache.values():
        if r["tour"] != tour_label:
            continue
        d = parse_ymd(r["date"].replace("-", ""))
        if not d:
            continue
        # Two different players can share an abbreviated name ("J. Smith"); keep them apart.
        def nm(n, k):
            owner = names.setdefault(n, k)
            return n if owner == k else f"{n} ({k})"
        w, l = nm(r["winner"], r["winner_key"]), nm(r["loser"], r["loser_key"])
        def srv(won, pts):
            won, pts = to_float(won), to_float(pts)
            return (won, pts) if won is not None and pts else None
        out.append({"date": d, "order": (r["time"], r["event_key"]), "surface": r["surface"] or None,
                    "winner": w, "loser": l, "status": r["status"], "tournament": r["tournament"],
                    "w_srv": srv(r["w_srv_won"], r["w_srv_pts"]), "l_srv": srv(r["l_srv_won"], r["l_srv_pts"])})
    return out


# ---------- source 2: Sackmann CSVs -------------------------------------------
def sackmann_status(score):
    s = (score or "").upper()
    if "W/O" in s or "WALKOVER" in s: return "wo"
    if "RET" in s or "DEF" in s or "ABD" in s or "ABN" in s: return "ret"
    return "completed"

def from_sackmann_rows(rows):
    out = []
    for i, r in enumerate(rows):
        d = parse_ymd(r.get("tourney_date"))
        w, l = (r.get("winner_name") or "").strip(), (r.get("loser_name") or "").strip()
        if not d or not w or not l:
            continue
        m = {"date": d, "order": (str(r.get("tourney_id", "")), to_float(r.get("match_num")) or i),
             "surface": norm_surface(r.get("surface")), "winner": w, "loser": l,
             "status": sackmann_status(r.get("score")), "tournament": r.get("tourney_name", "")}
        for who, key in (("w", "w_srv"), ("l", "l_srv")):
            sv, a, b = to_float(r.get(f"{who}_svpt")), to_float(r.get(f"{who}_1stWon")), to_float(r.get(f"{who}_2ndWon"))
            m[key] = (a + b, sv) if sv and sv > 0 and a is not None and b is not None else None
        out.append(m)
    return out

def fetch_sackmann(tour, years):
    cfg, rows = SACKMANN[tour], []
    for y in years:
        got, err = None, None
        for branch in ("master", "main"):
            try:
                text = http_get(RAW.format(repo=cfg["repo"], branch=branch, file=cfg["file"].format(year=y)))
                got = list(csv.DictReader(io.StringIO(text.decode("utf-8", "replace"))))
                break
            except Exception as e:                   # noqa: BLE001
                err = e
        if got:
            print(f"  · sackmann {tour} {y}: {len(got)} matches", file=sys.stderr)
            rows += got
        else:
            print(f"  · sackmann {tour} {y}: not available ({err})", file=sys.stderr)
    return from_sackmann_rows(rows)


# ---------- source 3: tennis-data.co.uk spreadsheets --------------------------
_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

def _col_index(ref):
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group(0):
        n = n * 26 + (ord(ch) - 64)
    return n - 1

def read_xlsx(data):
    """First worksheet of an .xlsx as a list of dicts keyed by the header row."""
    z = zipfile.ZipFile(io.BytesIO(data))
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", _NS):
            shared.append("".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")))
    sheets = sorted(n for n in z.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", n))
    root = ET.fromstring(z.read(sheets[0]))
    table = []
    for row in root.iter(f"{{{_NS['m']}}}row"):
        vals = {}
        for c in row.findall("m:c", _NS):
            t, v = c.get("t"), c.find("m:v", _NS)
            if t == "s" and v is not None:
                val = shared[int(v.text)]
            elif t == "inlineStr":
                val = "".join(x.text or "" for x in c.iter(f"{{{_NS['m']}}}t"))
            else:
                val = v.text if v is not None else ""
            vals[_col_index(c.get("r"))] = val
        if vals:
            table.append([vals.get(i, "") for i in range(max(vals) + 1)])
    if not table:
        return []
    head = [h.strip() for h in table[0]]
    return [dict(zip(head, r)) for r in table[1:]]

def td_name(n):
    """'Sinner J.' -> 'J. Sinner', 'De Minaur A.' -> 'A. De Minaur', 'Kwon S.W.' -> 'S.W. Kwon'."""
    n = (n or "").strip()
    m = re.match(r"^(.*?)\s+((?:[A-Za-z]{1,2}\.)(?:[- ]?[A-Za-z]{1,2}\.)*)$", n)
    return f"{m.group(2)} {m.group(1)}" if m else n

def td_date(v):
    f = to_float(v)
    if f is not None and f > 20000:                       # Excel serial number
        return date(1899, 12, 30) + timedelta(days=int(f))
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(str(v)[:10], fmt).date()
        except ValueError:
            pass
    return None

def from_tennis_data_rows(rows):
    out = []
    for i, r in enumerate(rows):
        d = td_date(r.get("Date"))
        w, l = td_name(r.get("Winner")), td_name(r.get("Loser"))
        if not d or not w or not l:
            continue
        c = (r.get("Comment") or "").lower()
        status = "wo" if "walk" in c else ("ret" if ("retir" in c or "disq" in c or "award" in c) else "completed")
        out.append({"date": d, "order": (r.get("Tournament", ""), i),
                    "surface": norm_surface(r.get("Surface")), "winner": w, "loser": l,
                    "status": status, "tournament": r.get("Tournament", ""),
                    "w_srv": None, "l_srv": None})
    return out

def fetch_tennis_data(tour, years):
    rows = []
    for y in years:
        try:
            got = read_xlsx(http_get(TENNIS_DATA[tour].format(year=y)))
            print(f"  · tennis-data {tour} {y}: {len(got)} matches", file=sys.stderr)
            rows += got
        except Exception as e:                       # noqa: BLE001
            print(f"  · tennis-data {tour} {y}: not available ({e})", file=sys.stderr)
    return from_tennis_data_rows(rows)


# ---------- computation (one tour) -------------------------------------------
def compute_tour(matches, tour_label, top_n, prev_players=None):
    """matches: normalised rows (see from_sackmann_rows). Returns (players, tour_avg, has_serve)."""
    matches = sorted((m for m in matches if m["surface"]), key=lambda m: (m["date"], m["order"]))
    if not matches:
        return {}, None, False
    latest = matches[-1]["date"]
    recent_cut = latest - timedelta(days=RECENT_DAYS)

    # --- ELO (walkovers and retirements skipped) ---
    elo_all, elo_surf, played, last_seen = {}, {}, {}, {}
    for m in matches:
        w, l, surf = m["winner"], m["loser"], m["surface"]
        last_seen[w] = last_seen[l] = m["date"]
        if m["status"] != "completed":
            continue
        rw, rl = elo_all.get(w, DEFAULT_ELO), elo_all.get(l, DEFAULT_ELO)
        kw, kl = k_factor(played.get(w, 0)), k_factor(played.get(l, 0))
        ew = expected(rw, rl)
        elo_all[w] = rw + kw * (1 - ew)
        elo_all[l] = rl - kl * (1 - ew)
        sw, sl = elo_surf.get((w, surf), DEFAULT_ELO), elo_surf.get((l, surf), DEFAULT_ELO)
        ews = expected(sw, sl)
        elo_surf[(w, surf)] = sw + kw * (1 - ews)
        elo_surf[(l, surf)] = sl - kl * (1 - ews)
        played[w] = played.get(w, 0) + 1
        played[l] = played.get(l, 0) + 1

    # --- serve/return: points-weighted, recency-weighted, opponent-adjusted ---
    # One entry per (server, returner, surface, weight, points won, points played).
    serve_obs = []
    for m in matches:
        if m["status"] == "wo":
            continue
        wt = 0.5 ** ((latest - m["date"]).days / HALF_LIFE_DAYS)
        if m["w_srv"]: serve_obs.append((m["winner"], m["loser"], m["surface"], wt) + m["w_srv"])
        if m["l_srv"]: serve_obs.append((m["loser"], m["winner"], m["surface"], wt) + m["l_srv"])
    has_serve = bool(serve_obs)

    tot_w = sum(o[3] * o[5] for o in serve_obs)
    T = (sum(o[3] * o[4] for o in serve_obs) / tot_w) if tot_w else (0.64 if tour_label == "ATP" else 0.56)

    # Opponent ratings: start from raw overall rates, then re-estimate each
    # player against their opponents' current ratings a few times so the
    # opponent ratings are themselves adjusted. Shrunk toward the tour average.
    S, R = {}, {}
    for _ in range(4):
        s_acc, r_acc = {}, {}
        for srv, ret, _, wt, won, pts in serve_obs:
            rate_ = won / pts
            a = s_acc.setdefault(srv, [0.0, 0.0]); a[0] += wt * pts * (rate_ + (R.get(ret, 1 - T) - (1 - T))); a[1] += wt * pts
            b = r_acc.setdefault(ret, [0.0, 0.0]); b[0] += wt * pts * ((1 - rate_) + (S.get(srv, T) - T)); b[1] += wt * pts
        S = {p: (v[0] + SHRINK_POINTS * T) / (v[1] + SHRINK_POINTS) for p, v in s_acc.items()}
        R = {p: (v[0] + SHRINK_POINTS * (1 - T)) / (v[1] + SHRINK_POINTS) for p, v in r_acc.items()}

    # per-surface adjusted rates against the final opponent ratings
    srv_acc, ret_acc = {}, {}      # key (player, surface|overall) -> [sum w*pts*adj_rate, sum w*pts, n matches]
    def acc(d, key, rate, weight):
        e = d.setdefault(key, [0.0, 0.0, 0]); e[0] += rate * weight; e[1] += weight; e[2] += 1
    for srv, ret, surf, wt, won, pts in serve_obs:
        rate = won / pts
        adj_s = rate + (R.get(ret, 1 - T) - (1 - T))       # tough returner -> serve rate counts for more
        adj_r = (1 - rate) + (S.get(srv, T) - T)           # big server -> return rate counts for more
        for k in (surf, "overall"):
            acc(srv_acc, (srv, k), adj_s, wt * pts)
            acc(ret_acc, (ret, k), adj_r, wt * pts)

    def rate(d, name, k):
        v = d.get((name, k))
        if v and v[2] >= MIN_SURFACE_MATCHES and v[1] > 0:
            return round(v[0] / v[1], 4)
        v = d.get((name, "overall"))
        return round(v[0] / v[1], 4) if v and v[1] > 0 else None

    # --- who gets listed ---
    candidates = [p for p in last_seen if last_seen[p] >= recent_cut] or list(last_seen)
    candidates.sort(key=lambda p: elo_all.get(p, DEFAULT_ELO), reverse=True)
    chosen = candidates if top_n <= 0 else candidates[:top_n]

    prev_by_loose = {}
    for p in (prev_players or {}).values():
        if p.get("tour") == tour_label:
            prev_by_loose.setdefault(loose_key(p["name"]), p)

    players = {}
    for name in chosen:
        elo = {"overall": round(elo_all.get(name, DEFAULT_ELO))}
        for s in SURFACES:
            if (name, s) in elo_surf:
                elo[s] = round(elo_surf[(name, s)])
        prev = prev_by_loose.get(loose_key(name))
        entry = {"name": name, "tour": tour_label, "elo": elo}
        if has_serve:
            serve = {"overall": rate(srv_acc, name, "overall")}
            rtn = {"overall": rate(ret_acc, name, "overall")}
            if serve["overall"] is None or rtn["overall"] is None:
                continue
            for s in SURFACES:
                sv, rt = rate(srv_acc, name, s), rate(ret_acc, name, s)
                if sv is not None: serve[s] = sv
                if rt is not None: rtn[s] = rt
            entry.update(serve=serve, **{"return": rtn})
        elif prev:
            # results-only source: keep last known serve/return, and the fuller display name
            entry.update(name=prev["name"], serve=prev["serve"], **{"return": prev["return"]})
            entry["stats_carried_over"] = True
        else:
            entry.update(serve={"overall": round(T, 4)}, **{"return": {"overall": round(1 - T, 4)}})
            entry["stats_estimated"] = True
        players[slugify(entry["name"])] = entry

    return players, round(T, 4), has_serve


def recent_results(matches, tour_label):
    if not matches:
        return []
    latest = max(m["date"] for m in matches)
    cut = latest - timedelta(days=RESULTS_DAYS)
    return [{"date": m["date"].isoformat(), "tour": tour_label, "tournament": m.get("tournament", ""),
             "winner": m["winner"], "loser": m["loser"], "status": m["status"]}
            for m in matches if m["date"] >= cut]


def load_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    yr = date.today().year
    ap.add_argument("--tours", nargs="+", default=["atp", "wta"], choices=["atp", "wta"])
    ap.add_argument("--years", nargs="+", type=int, default=[yr - 2, yr - 1, yr])
    ap.add_argument("--top", type=int, default=0, help="players kept PER TOUR (0 = all)")
    ap.add_argument("--out", default="players.json")
    ap.add_argument("--results-out", default="recent_results.json")
    ap.add_argument("--local", default=None, help="local Sackmann-format CSV, processed as the first --tours value")
    ap.add_argument("--local-xlsx", default=None, help="local tennis-data .xlsx, processed as the first --tours value")
    ap.add_argument("--api-cache", default="data/api_tennis_matches.csv")
    args = ap.parse_args()

    prev = load_json(args.out) or {}
    api_cache, api_err = None, None
    key = os.environ.get("TENNIS_API_KEY", "").strip()
    if key and not (args.local or args.local_xlsx):
        print("Syncing api-tennis.com…", file=sys.stderr)
        api_cache, n, api_err = sync_api_tennis(key, args.api_cache, args.tours)
        print(f"  · api-tennis: {len(api_cache)} matches cached, {n} new or updated", file=sys.stderr)
    elif os.path.exists(args.api_cache) and not (args.local or args.local_xlsx):
        api_cache = load_api_cache(args.api_cache)    # no key today: still use what's cached
    prev_players = prev.get("players", {})
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    players_all, tours_meta, results = {}, {}, []
    for tour in (args.tours[:1] if (args.local or args.local_xlsx) else args.tours):
        label = tour.upper()
        print(f"Loading {label} matches…", file=sys.stderr)
        if args.local:
            with open(args.local, encoding="utf-8") as fh:
                matches, source = from_sackmann_rows(list(csv.DictReader(fh))), "local csv"
        elif args.local_xlsx:
            with open(args.local_xlsx, "rb") as fh:
                matches, source = from_tennis_data_rows(read_xlsx(fh.read())), "local xlsx"
        else:
            matches, source = (from_api_cache(api_cache, label), "api-tennis.com") if api_cache else ([], None)
            if matches and api_err:
                source += " (last sync incomplete)"
            if not matches:
                matches, source = fetch_sackmann(tour, args.years), "sackmann"
            if not matches:
                print(f"  · {label}: Sackmann unavailable, trying tennis-data.co.uk", file=sys.stderr)
                matches, source = fetch_tennis_data(tour, args.years), "tennis-data.co.uk"

        pl, avg, has_serve = compute_tour(matches, label, args.top, prev_players)
        if pl:
            latest = max(m["date"] for m in matches).isoformat()
            players_all.update(pl)
            tours_meta[tour] = {"tour_avg_spw": avg if has_serve else
                                (prev.get("tours", {}).get(tour, {}).get("tour_avg_spw") or avg),
                                "source": source, "latest_match": latest,
                                "serve_stats": "fresh" if has_serve else "carried over"}
            if date.fromisoformat(latest) < date.today() - timedelta(days=STALE_AFTER_DAYS):
                tours_meta[tour]["stale"] = True      # a source answered, but with old results
            results += recent_results(matches, label)
        else:
            # nothing reachable: keep yesterday's players for this tour, marked stale
            kept = {k: v for k, v in prev_players.items() if v.get("tour") == label}
            print(f"  · {label}: no source reachable — keeping {len(kept)} players from the previous file",
                  file=sys.stderr)
            players_all.update(kept)
            if kept:
                old = dict(prev.get("tours", {}).get(tour, {}))
                old["stale"] = True
                old.setdefault("source", "previous file")
                tours_meta[tour] = old

    if not players_all:
        sys.exit("No data produced and no previous players.json to keep. Check network or pass --local <csv>.")

    stale = [t for t, m in tours_meta.items() if m.get("stale")]
    out = {
        "updated": now if len(stale) < len(tours_meta) else prev.get("updated", now),
        "checked": now,
        "stale": stale,
        "source": " · ".join(f"{t}: {m.get('source')}" for t, m in tours_meta.items()),
        "tours": tours_meta,
        "tour_avg_spw": tours_meta.get("atp", {}).get("tour_avg_spw", 0.64),  # legacy default
        "players": players_all,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    if results:
        with open(args.results_out, "w", encoding="utf-8") as fh:
            json.dump({"updated": now, "results": sorted(results, key=lambda r: r["date"])},
                      fh, ensure_ascii=False, indent=1)
    by_tour = {t: sum(1 for p in players_all.values() if p["tour"] == t.upper()) for t in tours_meta}
    print(f"Wrote {args.out}: {len(players_all)} players {by_tour}, stale={stale}; "
          f"{len(results)} recent results", file=sys.stderr)


if __name__ == "__main__":
    main()
