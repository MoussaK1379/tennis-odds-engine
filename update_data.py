"""Daily data updater for the Tennis Odds Engine (ATP + WTA).

Pulls recent ATP and WTA match results, computes current surface ELO and
serve/return rates per player, and writes players.json — the file the web app
reads on load. Each player is tagged with their tour, and each tour carries its
own average serve-points-won baseline (WTA serve hold rates run lower than ATP),
so the Markov math uses the right reference for each.

Sources, tried in order for each tour:
  1. api-tennis.com (needs TENNIS_API_KEY) — results plus serve points won.
     Matches are cached in data/api_tennis_matches.csv: the first run
     backfills about two years, later runs fetch the last week and, with any
     time left, extend the history back to API_HISTORY_FROM a few weeks at a
     time. The cache keeps the history even if the API stops responding.
  2. Jeff Sackmann's match CSVs on GitHub — results plus serve/return stats.
  3. tennis-data.co.uk yearly spreadsheets — results only. ELO stays current;
     serve/return rates are carried over from the previous players.json
     (new players get the tour average) and flagged as carried over.
  4. None reachable — the previous players.json is kept as-is and marked
     stale, so the site keeps working and says the data is old.

Also writes recent_results.json (the last few weeks of results), which
update_matches.py uses to settle its logged predictions.

The ratings themselves come from model.py (shared with the match list and the
backtest), using the settings in model_params.json that tools/backtest.py
chose. A player is listed if they played in the last 12 months.

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
RECENT_DAYS = 365          # listed if played within this many days of the latest result
RESULTS_DAYS = 28          # how much history recent_results.json keeps
STALE_AFTER_DAYS = 14      # newest result older than this -> tour marked stale
PARAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_params.json")


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
API_TENNIS_TYPES = {"atp": ("265", "Atp Singles"), "wta": ("266", "Wta Singles"),
                    # Challenger level (ATP Challenger, WTA 125), cached as tour ATP_CH / WTA_CH
                    "atp_ch": ("281", "Challenger Men Singles"), "wta_ch": ("272", "Challenger Women Singles")}
CHALLENGER_OF = {"ATP": "ATP_CH", "WTA": "WTA_CH"}
API_CACHE_FIELDS = ["event_key", "date", "time", "tour", "tournament", "round", "qualifying",
                    "surface", "indoor", "status", "winner", "winner_key", "loser", "loser_key",
                    "w_srv_won", "w_srv_pts", "l_srv_won", "l_srv_pts",
                    "w_sets", "l_sets", "games"]
API_STATUS = {"Finished": "completed", "Retired": "ret", "Walk Over": "wo"}
API_WINDOW_DAYS = 7        # date range per request
API_REFRESH_DAYS = 7       # re-fetch this many recent days each run (late stats, corrections)
API_BACKFILL_DAYS = 730    # history fetched on the first run
API_HISTORY_FROM = date(2022, 1, 1)   # older history, fetched backwards with spare time each run
API_HISTORY_EMPTY = 8      # this many empty weeks in a row = the API has nothing older
API_BUDGET_S = 18 * 60     # stop fetching history after this long; the next run carries on
API_SAVE_EVERY = 8         # save the cache every this many requests, so a killed run keeps its work
SURFACES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tournament_surfaces.json")

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
    # sets and games from the winner's side: "1 - 2" is first player - second player
    w_first = winner_side == "First Player"
    sets = re.match(r"\s*(\d+)\s*-\s*(\d+)", ev.get("event_final_result") or "")
    w_sets = l_sets = ""
    if sets:
        f_, s_ = int(sets.group(1)), int(sets.group(2))
        w_sets, l_sets = (f_, s_) if w_first else (s_, f_)
    games = []
    for sc in sorted(ev.get("scores") or [], key=lambda x: to_float(x.get("score_set")) or 0):
        a_, b_ = sc.get("score_first"), sc.get("score_second")
        if a_ not in (None, "") and b_ not in (None, ""):
            games.append(f"{a_}-{b_}" if w_first else f"{b_}-{a_}")
    return {"event_key": ev.get("event_key"), "date": ev.get("event_date"), "time": ev.get("event_time") or "",
            "tour": tour.upper(), "tournament": ev.get("tournament_name", ""),
            "round": ev.get("tournament_round", ""), "qualifying": ev.get("event_qualification") == "True",
            "surface": surf, "indoor": indoor, "status": status,
            "winner": w.strip(), "winner_key": wk, "loser": l.strip(), "loser_key": lk,
            "w_srv_won": ws[0] if ws else "", "w_srv_pts": ws[1] if ws else "",
            "l_srv_won": ls[0] if ls else "", "l_srv_pts": ls[1] if ls else "",
            "w_sets": w_sets, "l_sets": l_sets, "games": " ".join(games)}

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

def _state_path(cache_path):
    return os.path.join(os.path.dirname(cache_path) or ".", "api_tennis_state.json")

def sync_api_tennis(key, cache_path, tours, today=None, budget_s=API_BUDGET_S):
    """Bring the cache up to date. Returns (cache, number of new/updated rows, error or None).

    A new feed (e.g. the Challengers) is backfilled API_BACKFILL_DAYS; the backfill stops
    after budget_s seconds and the next run carries on from the newest cached match.
    Matches cached before set scores were stored are re-fetched once (a one-off
    backfill tracked in api_tennis_state.json, resumable if it's interrupted).
    Time left after the recent days goes to older history (see extend_history)."""
    import time
    t0 = time.time()
    today = today or date.today()
    cache = load_api_cache(cache_path)
    state = load_json(_state_path(cache_path)) or {}
    sets_state = state.setdefault("sets_backfill", {})
    changed, err, calls = 0, None, 0
    def save():
        save_api_cache(cache_path, cache)
        with open(_state_path(cache_path), "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=1)
    def fetch(tour, a, b):
        """Cache one window of results. -> number of matches in it (raises on API errors)."""
        nonlocal changed, calls
        type_key, type_name = API_TENNIS_TYPES[tour]
        events = api_tennis_call(key, method="get_fixtures", event_type_key=type_key,
                                 date_start=a.isoformat(), date_stop=b.isoformat())
        n = 0
        for ev in events:
            if ev.get("event_type_type") != type_name:
                continue
            row = api_event_row(ev, tour)
            if row:
                n += 1
                k = str(row["event_key"])
                row = {f: ("" if row[f] is None else str(row[f])) for f in API_CACHE_FIELDS}
                if cache.get(k) != row:
                    cache[k] = row
                    changed += 1
        calls += 1
        if calls % API_SAVE_EVERY == 0:
            save()
        return n
    for tour in tours:
        # Per tour, so an interrupted backfill of one tour resumes where it stopped.
        have = [r["date"] for r in cache.values() if r["tour"] == tour.upper()]
        if not have:
            sets_state[tour] = "done"      # a new feed: its rows carry set scores from the start
            print(f"  · api-tennis {tour}: new feed, fetching {API_BACKFILL_DAYS} days of history", file=sys.stderr)
        a = (datetime.strptime(max(have), "%Y-%m-%d").date() - timedelta(days=API_REFRESH_DAYS)
             if have else today - timedelta(days=API_BACKFILL_DAYS))
        backfill = bool(have) and sets_state.get(tour) != "done"
        if backfill:                       # re-fetch the history once to add set scores
            done_to = sets_state.get(tour)
            a = (datetime.strptime(done_to, "%Y-%m-%d").date() + timedelta(days=1)
                 if done_to else datetime.strptime(min(have), "%Y-%m-%d").date())
            print(f"  · api-tennis {tour}: adding set scores to history from {a}", file=sys.stderr)
        while a <= today:
            if a < today - timedelta(days=API_REFRESH_DAYS) and time.time() - t0 > budget_s:
                print(f"  · api-tennis {tour}: time budget used, history fetched up to {a - timedelta(days=1)};"
                      " the next run carries on", file=sys.stderr)
                break
            b = min(a + timedelta(days=API_WINDOW_DAYS - 1), today)
            try:
                fetch(tour, a, b)
            except Exception as e:                   # noqa: BLE001
                err = f"{tour} {a}..{b}: {e}"
                print(f"  · api-tennis {err}", file=sys.stderr)
                break                                # keep what we have; retry tomorrow
            if backfill:
                sets_state[tour] = b.isoformat()
            a = b + timedelta(days=1)
        else:
            if backfill:
                sets_state[tour] = "done"
    if not err:
        extend_history(cache, state.setdefault("history", {}), tours, fetch,
                       lambda: time.time() - t0 > budget_s)
    save()
    return cache, changed, err


def extend_history(cache, hist, tours, fetch, out_of_time):
    """Fetch older results backwards from each feed's earliest cached match until
    API_HISTORY_FROM, a week per call, while there's time. hist (saved in the state
    file) records per feed how far back it got, how many empty weeks in a row it has
    seen, or "done"; a long run of empty weeks means the API keeps nothing older."""
    for tour in tours:
        h = hist.get(tour)
        if h == "done":
            continue
        have = [r["date"] for r in cache.values() if r["tour"] == tour.upper()]
        if not have:
            continue
        h = h or {}
        b = (datetime.strptime(h["from"], "%Y-%m-%d").date() if h.get("from")
             else datetime.strptime(min(have), "%Y-%m-%d").date()) - timedelta(days=1)
        empty = h.get("empty", 0)
        while b >= API_HISTORY_FROM and empty < API_HISTORY_EMPTY and not out_of_time():
            a = max(b - timedelta(days=API_WINDOW_DAYS - 1), API_HISTORY_FROM)
            try:
                n = fetch(tour, a, b)
            except Exception as e:                   # noqa: BLE001
                print(f"  · api-tennis {tour} history {a}..{b}: {e}", file=sys.stderr)
                break                                # try again next run
            empty = 0 if n else empty + 1
            h = hist[tour] = {"from": a.isoformat(), "empty": empty}
            b = a - timedelta(days=1)
        if b < API_HISTORY_FROM or empty >= API_HISTORY_EMPTY:
            print(f"  · api-tennis {tour}: history complete back to {h.get('from', min(have))}", file=sys.stderr)
            hist[tour] = "done"
        else:
            if h.get("from"):
                print(f"  · api-tennis {tour}: history fetched back to {h['from']}; the next run carries on",
                      file=sys.stderr)
            hist[tour] = h

def sync_tournament_surfaces(key, path=SURFACES_PATH):
    """api-tennis's own surface for each tournament, saved as {feed: {tournament name: surface}}
    for the Challenger events the tournament table doesn't list. Returns a short report."""
    try:
        res = api_tennis_call(key, method="get_tournaments")
    except Exception as e:                           # noqa: BLE001
        return f"tournament list unavailable ({e})"
    if not isinstance(res, list):
        return f"tournament list: unexpected reply ({type(res).__name__})"
    by_type = {v[1]: k.upper() for k, v in API_TENNIS_TYPES.items()}
    out, fields, with_surf = {}, set(), 0
    for t in res:
        if not isinstance(t, dict):
            continue
        fields.update(t)
        feed = by_type.get(t.get("event_type_type"))
        raw = str(t.get("tournament_sourface") or t.get("tournament_surface") or t.get("surface") or "").lower()
        surf = next((x for x in ("clay", "grass", "hard") if x in raw), "hard" if "carpet" in raw else None)
        if feed and t.get("tournament_name") and surf:
            out.setdefault(feed, {})[t["tournament_name"]] = surf
            with_surf += 1
    # how often api-tennis agrees with the tournament table on tour events (a trust check)
    from update_matches import _match_event
    agree = total = 0
    for feed in ("ATP", "WTA"):
        for name, surf in (out.get(feed) or {}).items():
            hit = _match_event(name, feed)
            if hit:
                total += 1; agree += hit[1] == surf
    if out:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1, sort_keys=True)
    counts = {k: len(v) for k, v in out.items()}
    return (f"{len(res)} tournaments, {with_surf} with a surface {counts}; agrees with the table on "
            f"{agree}/{total} tour events" + ("" if with_surf else f"; fields: {sorted(fields)[:12]}"))

def _first_set(games):
    """'6-4 3-6 7-6' (winner's side) -> (6, 4); None if missing or not a finished set.
    api-tennis writes tiebreak sets with the tiebreak points after a dot: '7.7-6.5' is 7-6."""
    m = re.match(r"\s*(\d+)(?:\.\d+)?-(\d+)(?:\.\d+)?", games or "")
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    done = (max(a, b) == 6 and abs(a - b) >= 2) or (max(a, b) == 7 and min(a, b) in (5, 6))
    return (a, b) if done else None

def from_api_cache(cache, tour_label, challenger=False):
    """Cached matches for one tour as model rows. challenger=True also includes that tour's
    Challenger level (ATP Challenger / WTA 125), marked level 'challenger'."""
    # Surface, court and format come from the tournament table at load time, so
    # fixes to the table also apply to matches cached before the fix.
    from update_matches import surface_for, court_id, guess_best_of
    feeds = (tour_label, CHALLENGER_OF[tour_label]) if challenger and tour_label in CHALLENGER_OF else (tour_label,)
    rows = [r for r in cache.values() if r["tour"] in feeds]
    # Two different players can share an abbreviated name ("J. Smith"); keep them apart.
    # Tour-level players claim a name first, so adding Challengers never renames them.
    names = {}
    for r in sorted(rows, key=lambda r: (r["tour"] != tour_label, r["date"])):
        for n, k in ((r["winner"], r["winner_key"]), (r["loser"], r["loser_key"])):
            names.setdefault(n, k)
    def nm(n, k):
        return n if names.get(n) == k else f"{n} ({k})"
    out = []
    for r in rows:
        d = parse_ymd(r["date"].replace("-", ""))
        if not d:
            continue
        w, l = nm(r["winner"], r["winner_key"]), nm(r["loser"], r["loser_key"])
        def srv(won, pts):
            won, pts = to_float(won), to_float(pts)
            return (won, pts) if won is not None and pts else None
        feed = r["tour"]
        surf, _, _ = surface_for(r["tournament"], feed, d.month)
        qual = r.get("qualifying") == "True"
        ch = feed != tour_label
        out.append({"date": d, "order": (r["time"], r["event_key"]), "surface": surf,
                    "winner": w, "loser": l, "status": r["status"], "tournament": r["tournament"],
                    "court": court_id(r["tournament"], feed, d.month),
                    "set1": _first_set(r.get("games")),
                    "sets": ((int(r["w_sets"]), int(r["l_sets"]))
                             if (r.get("w_sets") or "").isdigit() and (r.get("l_sets") or "").isdigit() else None),
                    "best_of": 3 if qual or ch else guess_best_of(r["tournament"], tour_label),
                    "qualifying": qual, "level": "challenger" if ch else "tour",
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
def load_params(path=PARAMS_PATH):
    """Model settings chosen by tools/backtest.py --tune (defaults if the file is missing)."""
    import model
    try:
        with open(path) as fh:
            return model.Params.from_dict(json.load(fh)["params"])
    except (OSError, ValueError, KeyError):
        return model.Params()

def params_version(path=PARAMS_PATH):
    try:
        with open(path) as fh:
            return json.load(fh).get("version", "")
    except (OSError, ValueError):
        return ""


def compute_tour(matches, tour_label, top_n, prev_players=None, params=None, today=None):
    """matches: normalised rows (see from_sackmann_rows). Runs them through the
    shared model in date order. Returns (players, tour_avg, has_serve, model_block)."""
    import model
    P = params or load_params()
    matches = sorted((m for m in matches if m["surface"]), key=lambda m: (m["date"], m["order"]))
    if not matches:
        return {}, None, False, None
    latest = matches[-1]["date"]
    as_of = max(today or latest, latest)
    recent_cut = as_of - timedelta(days=RECENT_DAYS)

    eng = model.TourEngine(tour_label, P)
    for m in matches:
        eng.update(m)
    has_serve = eng.T_den > 0
    T = eng.T()

    last_seen = {p: max(i["dates"]) for p, i in eng.info.items() if i["dates"]}
    candidates = [p for p in last_seen if last_seen[p] >= recent_cut] or list(last_seen)
    candidates.sort(key=lambda p: eng.rating(p, None, as_of)[0], reverse=True)
    chosen = candidates if top_n <= 0 else candidates[:top_n]

    prev_by_loose = {}
    for p in (prev_players or {}).values():
        if p.get("tour") == tour_label:
            prev_by_loose.setdefault(loose_key(p["name"]), p)

    players = {}
    for name in chosen:
        entry = {"name": name, "tour": tour_label, **eng.export(name, as_of)}
        if not has_serve:
            prev = prev_by_loose.get(loose_key(name))
            if prev:
                # results-only source: keep last known serve/return, and the fuller display name
                entry.update(name=prev["name"], serve=prev["serve"], **{"return": prev["return"]})
                entry["n"] = prev.get("n", 0)
                entry["stats_carried_over"] = True
            else:
                entry["stats_estimated"] = True
        players[slugify(entry["name"])] = entry

    return players, round(T, 4), has_serve, eng.model_block()


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
    ap.add_argument("--api-feeds", nargs="+", default=list(API_TENNIS_TYPES), choices=list(API_TENNIS_TYPES),
                    help="api-tennis feeds kept in the cache (Challengers are fetched even while the model doesn't use them yet)")
    args = ap.parse_args()

    prev = load_json(args.out) or {}
    api_cache, api_err = None, None
    key = os.environ.get("TENNIS_API_KEY", "").strip()
    if key and not (args.local or args.local_xlsx):
        print("Syncing api-tennis.com…", file=sys.stderr)
        api_cache, n, api_err = sync_api_tennis(key, args.api_cache, args.api_feeds)
        by_feed = {}
        for r in api_cache.values():
            by_feed[r["tour"]] = by_feed.get(r["tour"], 0) + 1
        print(f"  · api-tennis: {len(api_cache)} matches cached {by_feed}, {n} new or updated", file=sys.stderr)
        if any(f.endswith("_ch") for f in args.api_feeds):
            print(f"  · api-tennis surfaces: {sync_tournament_surfaces(key)}", file=sys.stderr)
    elif os.path.exists(args.api_cache) and not (args.local or args.local_xlsx):
        api_cache = load_api_cache(args.api_cache)    # no key today: still use what's cached
    prev_players = prev.get("players", {})
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    players_all, tours_meta, results = {}, {}, []
    params = load_params()
    model_block = prev.get("model") or {"rating": "elo", "w": {"ATP": 0.5, "WTA": 0.5}, "n0": 0}
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
            matches, source = ((from_api_cache(api_cache, label, challenger=params.ch_use), "api-tennis.com")
                               if api_cache else ([], None))
            if params.ch_use and matches:
                source += " incl. Challengers"
            if matches and api_err:
                source += " (last sync incomplete)"
            if not matches:
                matches, source = fetch_sackmann(tour, args.years), "sackmann"
            if not matches:
                print(f"  · {label}: Sackmann unavailable, trying tennis-data.co.uk", file=sys.stderr)
                matches, source = fetch_tennis_data(tour, args.years), "tennis-data.co.uk"

        pl, avg, has_serve, mblock = compute_tour(matches, label, args.top, prev_players, params,
                                                  today=date.today())
        if mblock:
            model_block = {**mblock, "version": params_version()}
        if pl:
            latest = max(m["date"] for m in matches).isoformat()
            for k, v in pl.items():
                # an ATP and a WTA player can share a name ("S. Johnson"): the second tour's
                # id gets a suffix instead of overwriting the first
                if k in players_all and players_all[k].get("tour") != label:
                    k = f"{k}-{label.lower()}"
                players_all[k] = v
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
        "model": model_block,
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
