"""Daily data updater for the Tennis Odds Engine (ATP + WTA).

Pulls recent ATP and WTA match results, computes current surface ELO and
serve/return rates per player, and writes players.json — the file the web app
reads on load. Each player is tagged with their tour, and each tour carries its
own average serve-points-won baseline (WTA serve hold rates run lower than ATP),
so the Markov math uses the right reference for each.

Self-contained (standard library only) so it runs cleanly on a CI runner.

Usage:
    python update_data.py                      # ATP + WTA, last 3 seasons
    python update_data.py --tours atp          # one tour only
    python update_data.py --years 2024 2025 2026 --top 60
    python update_data.py --local sample.csv --tours wta   # test a local CSV as one tour
"""

import argparse, csv, io, json, re, sys, unicodedata
import urllib.request
from datetime import date, datetime, timezone

TOURS = {
    "atp": {"repo": "JeffSackmann/tennis_atp", "file": "atp_matches_{year}.csv"},
    "wta": {"repo": "JeffSackmann/tennis_wta", "file": "wta_matches_{year}.csv"},
}
RAW = "https://raw.githubusercontent.com/{repo}/{branch}/{file}"
SURFACES = ("hard", "clay", "grass")
DEFAULT_ELO = 1500.0
MIN_SURFACE_MATCHES = 3


# ---------- tiny ELO ----------------------------------------------------------
def k_factor(m): return 250.0 / (m + 5) ** 0.4
def expected(ra, rb): return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


# ---------- helpers -----------------------------------------------------------
def slugify(name):
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", n.lower()).strip("-")

def norm_surface(s):
    s = (s or "").strip().lower()
    if s in ("hard", "clay", "grass"):
        return s
    return "hard" if s == "carpet" else None

def pts_won_serving(r, who):
    sv, a, b = r.get(f"{who}_svpt"), r.get(f"{who}_1stWon"), r.get(f"{who}_2ndWon")
    try:
        sv, a, b = float(sv), float(a), float(b)
        if sv > 0:
            return (a + b) / sv
    except (TypeError, ValueError):
        pass
    return None


# ---------- data source -------------------------------------------------------
def fetch_year(tour, year):
    cfg = TOURS[tour]
    last_err = None
    for branch in ("master", "main"):
        url = RAW.format(repo=cfg["repo"], branch=branch, file=cfg["file"].format(year=year))
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return list(csv.DictReader(io.StringIO(resp.read().decode("utf-8", "replace"))))
        except Exception as e:                       # noqa: BLE001
            last_err = e
    print(f"  · {tour} {year}: not available ({last_err})", file=sys.stderr)
    return []

def load_local(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------- computation (one tour) -------------------------------------------
def compute_tour(rows, tour_label, top_n, min_recent):
    rows = sorted(rows, key=lambda r: (r.get("tourney_date", ""), r.get("match_num", "")))

    elo_all, elo_surf, played = {}, {}, {}
    srv, ret = {}, {}
    recent_year = max((r.get("tourney_date", "0000")[:4] for r in rows), default="0000")
    recent_matches = {}
    serve_sum = serve_n = 0.0

    def acc(d, key, val):
        if val is None: return
        d.setdefault(key, [0.0, 0])
        d[key][0] += val; d[key][1] += 1

    for r in rows:
        surf = norm_surface(r.get("surface"))
        w, l = r.get("winner_name"), r.get("loser_name")
        if not surf or not w or not l:
            continue

        rw, rl = elo_all.get(w, DEFAULT_ELO), elo_all.get(l, DEFAULT_ELO)
        kw, kl = k_factor(played.get(w, 0)), k_factor(played.get(l, 0))
        ew = expected(rw, rl)
        elo_all[w] = rw + kw * (1 - ew)
        elo_all[l] = rl + kl * (0 - (1 - ew))

        sw, sl = elo_surf.get((w, surf), DEFAULT_ELO), elo_surf.get((l, surf), DEFAULT_ELO)
        ews = expected(sw, sl)
        elo_surf[(w, surf)] = sw + kw * (1 - ews)
        elo_surf[(l, surf)] = sl + kl * (0 - (1 - ews))

        played[w] = played.get(w, 0) + 1
        played[l] = played.get(l, 0) + 1
        if r.get("tourney_date", "")[:4] == recent_year:
            recent_matches[w] = recent_matches.get(w, 0) + 1
            recent_matches[l] = recent_matches.get(l, 0) + 1

        w_serve, l_serve = pts_won_serving(r, "w"), pts_won_serving(r, "l")
        if w_serve is not None:
            acc(srv, (w, surf), w_serve); acc(srv, (w, "overall"), w_serve)
            acc(ret, (l, surf), 1 - w_serve); acc(ret, (l, "overall"), 1 - w_serve)
            serve_sum += w_serve; serve_n += 1
        if l_serve is not None:
            acc(srv, (l, surf), l_serve); acc(srv, (l, "overall"), l_serve)
            acc(ret, (w, surf), 1 - l_serve); acc(ret, (w, "overall"), 1 - l_serve)
            serve_sum += l_serve; serve_n += 1

    candidates = [p for p in elo_all if recent_matches.get(p, 0) >= min_recent] or list(elo_all)
    candidates.sort(key=lambda p: (recent_matches.get(p, 0), elo_all[p]), reverse=True)
    chosen = candidates if top_n <= 0 else candidates[:top_n]
    chosen = sorted(chosen, key=lambda p: elo_all[p], reverse=True)

    def rate(d, key, fb_key):
        v = d.get(key)
        if v and v[1] >= MIN_SURFACE_MATCHES:
            return round(v[0] / v[1], 4)
        fb = d.get(fb_key)
        return round(fb[0] / fb[1], 4) if fb and fb[1] else None

    players = {}
    for name in chosen:
        elo = {"overall": round(elo_all[name])}
        serve = {"overall": rate(srv, (name, "overall"), (name, "overall"))}
        rtn = {"overall": rate(ret, (name, "overall"), (name, "overall"))}
        for s in SURFACES:
            if (name, s) in elo_surf:
                elo[s] = round(elo_surf[(name, s)])
            sv = rate(srv, (name, s), (name, "overall"))
            rt = rate(ret, (name, s), (name, "overall"))
            if sv is not None: serve[s] = sv
            if rt is not None: rtn[s] = rt
        if serve["overall"] is None:
            continue
        players[slugify(name)] = {"name": name, "tour": tour_label,
                                  "elo": elo, "serve": serve, "return": rtn}

    tour_avg = round(serve_sum / serve_n, 4) if serve_n else 0.60
    return players, tour_avg


def main():
    ap = argparse.ArgumentParser()
    yr = date.today().year
    ap.add_argument("--tours", nargs="+", default=["atp", "wta"], choices=["atp", "wta"])
    ap.add_argument("--years", nargs="+", type=int, default=[yr - 2, yr - 1, yr])
    ap.add_argument("--top", type=int, default=0, help="players kept PER TOUR (0 = all)")
    ap.add_argument("--min-recent", type=int, default=1, help="min matches in the latest season to be included")
    ap.add_argument("--out", default="players.json")
    ap.add_argument("--local", default=None, help="local CSV, processed as the first --tours value")
    args = ap.parse_args()

    players_all, tours_meta = {}, {}

    if args.local:
        label = args.tours[0]
        print(f"Loading local {args.local} as {label.upper()}…", file=sys.stderr)
        pl, avg = compute_tour(load_local(args.local), label.upper(), args.top, args.min_recent)
        players_all.update(pl); tours_meta[label] = {"tour_avg_spw": avg}
    else:
        for tour in args.tours:
            print(f"Loading {tour.upper()} matches…", file=sys.stderr)
            rows = []
            for y in args.years:
                got = fetch_year(tour, y)
                if got:
                    print(f"  · {tour} {y}: {len(got)} matches", file=sys.stderr)
                    rows += got
            if not rows:
                print(f"  · {tour.upper()}: no data, skipping", file=sys.stderr)
                continue
            pl, avg = compute_tour(rows, tour.upper(), args.top, args.min_recent)
            players_all.update(pl); tours_meta[tour] = {"tour_avg_spw": avg}

    if not players_all:
        sys.exit("No data produced. Check network or pass --local <csv>.")

    out = {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "source": f"github.com/JeffSackmann · tours {list(tours_meta)} · years {args.years}",
        "tours": tours_meta,
        "tour_avg_spw": tours_meta.get("atp", {}).get("tour_avg_spw", 0.64),  # legacy default
        "players": players_all,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    by_tour = {t: sum(1 for p in players_all.values() if p["tour"] == t.upper()) for t in tours_meta}
    print(f"Wrote {args.out}: {len(players_all)} players {by_tour}, tours={tours_meta}", file=sys.stderr)


if __name__ == "__main__":
    main()
