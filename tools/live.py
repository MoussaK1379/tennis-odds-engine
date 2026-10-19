"""Live scores and up-to-the-minute start times for the site, from api-tennis.com.

Runs for hours inside .github/workflows/live.yml. Every LIVE_EVERY seconds it reads the
live scores (get_livescore); every FIXTURES_EVERY seconds it reads the day's fixtures
(get_fixtures, yesterday to tomorrow) for start times and statuses. It writes live.json:

  live    the ATP / WTA / Challenger singles under way: set scores, game score, server,
          status ("Set 2", "Interrupted" when play is suspended)
  events  every match from 12 hours ago to 36 hours ahead that isn't long over: its start
          time now, the start time first seen (so a match moved to a new time shows both),
          its status, and the result once it's finished

and publishes it on the "live" branch as a single commit replaced each time, so the site
can read it every 30 seconds from raw.githubusercontent.com without a site rebuild and
without filling the main history.

    TENNIS_API_KEY=... python tools/live.py --minutes 5 --no-publish    # try it locally
"""

import argparse, json, os, subprocess, sys, time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import update_data as u                           # noqa: E402
import update_matches as um                       # noqa: E402

LIVE_EVERY = int(os.environ.get("LIVE_EVERY", "60"))
FIXTURES_EVERY = int(os.environ.get("FIXTURES_EVERY", "600"))
TYPES = {name: tour.upper() for tour, (_, name) in u.API_TENNIS_TYPES.items()}   # "Atp Singles" -> "ATP"
KEEP_FINISHED_H = 3          # finished matches stay in events this long (the site hides them)


def iso(d):
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


def start_of(ev):
    try:
        return datetime.strptime(f"{ev.get('event_date')} {ev.get('event_time') or '00:00'}",
                                 "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def describe(ev, feed):
    """The fields the site needs for one api-tennis event."""
    st = start_of(ev)
    tour, level = feed[:3], ("challenger" if feed.endswith("_CH") else "tour")
    surf, indoor, _ = um.surface_for(ev.get("tournament_name", ""), feed, st.month if st else None)
    qual = um.is_qualifying(ev.get("event_qualification") == "True", ev.get("tournament_name"),
                            ev.get("tournament_round"))
    sets = []
    for sc in sorted(ev.get("scores") or [], key=lambda x: u.to_float(x.get("score_set")) or 0):
        a, b = u.to_float(sc.get("score_first")), u.to_float(sc.get("score_second"))
        if a is not None and b is not None:
            sets.append([int(a), int(b)])
    won = [0, 0]
    fr = (ev.get("event_final_result") or "").replace(" ", "")
    if "-" in fr and fr != "-":
        try:
            won = [int(x) for x in fr.split("-")[:2]]
        except ValueError:
            pass
    return {"k": str(ev.get("event_key")), "tour": tour, "level": level, "feed": feed,
            "tournament": ev.get("tournament_name", ""), "round": ev.get("tournament_round", ""),
            "qualifying": qual, "surface": surf, "indoor": indoor,
            "best_of": 3 if qual or level == "challenger" else um.guess_best_of(ev.get("tournament_name", ""), tour),
            "start": iso(st) if st else None, "status": (ev.get("event_status") or "").strip(),
            "first": (ev.get("event_first_player") or "").strip(), "first_key": ev.get("first_player_key"),
            "second": (ev.get("event_second_player") or "").strip(), "second_key": ev.get("second_player_key"),
            "sets": sets, "sets_won": won, "game": (ev.get("event_game_result") or "").replace(" ", ""),
            "serve": {"First Player": "first", "Second Player": "second"}.get(ev.get("event_serve")),
            "winner": {"First Player": "first", "Second Player": "second"}.get(ev.get("event_winner"))}


def is_live(e):
    s = e["status"].lower()
    return s.startswith("set ") or s == "interrupted" or "tie" in s


def is_over(e):
    return e["status"].lower() in ("finished", "retired", "walk over", "walkover", "cancelled", "canceled",
                                   "awarded", "defaulted", "abandoned")


class Live:
    def __init__(self, key, prev=None):
        self.key = key
        prev = prev or {}
        self.events = prev.get("events", {})          # k -> event dict (with first_start, ended)
        self.live = []
        self.errors = []

    def call(self, **params):
        return um._api_retry(lambda: u.api_tennis_call(self.key, timezone="Etc/UTC", **params))

    def fixtures(self, now):
        a, b = (now - timedelta(days=1)).date().isoformat(), (now + timedelta(days=1)).date().isoformat()
        for tour, (type_key, type_name) in u.API_TENNIS_TYPES.items():
            try:
                evs = self.call(method="get_fixtures", event_type_key=type_key, date_start=a, date_stop=b)
            except Exception as e:                       # noqa: BLE001
                self.errors.append(f"fixtures {tour}: {str(e).replace(self.key, '***')[:150]}")
                continue
            for ev in evs:
                if ev.get("event_type_type") == type_name:
                    self.merge(describe(ev, tour.upper()), now)

    def livescore(self, now):
        """-> True if a match left the live feed (finished, most likely: re-read the fixtures)."""
        try:
            evs = self.call(method="get_livescore")
        except Exception as e:                           # noqa: BLE001
            self.errors.append(f"livescore: {str(e).replace(self.key, '***')[:150]}")
            return False
        live = []
        for ev in evs if isinstance(evs, list) else []:
            feed = TYPES.get(ev.get("event_type_type"))
            if feed:
                d = self.merge(describe(ev, feed), now)
                if is_live(d):
                    live.append(d)
        gone = {d["k"] for d in self.live} - {d["k"] for d in live}
        self.live = sorted(live, key=lambda d: (d["tour"], d["level"], d["tournament"], d["start"] or ""))
        return bool(gone)

    def merge(self, d, now):
        old = self.events.get(d["k"], {})
        d["first_start"] = old.get("first_start") or d["start"]
        if is_over(d):
            d["ended"] = old.get("ended") or iso(now)
        self.events[d["k"]] = d
        return d

    def snapshot(self, now):
        keep, live_keys = {}, {d["k"] for d in self.live}
        for k, e in self.events.items():
            st = um.parse_iso(e.get("start"))
            if k not in live_keys and (not st or not (now - timedelta(hours=12) <= st <= now + timedelta(hours=36))):
                continue
            if e.get("ended") and um.parse_iso(e["ended"]) < now - timedelta(hours=KEEP_FINISHED_H):
                continue
            keep[k] = {f: e.get(f) for f in ("k", "tour", "level", "tournament", "round", "qualifying", "surface",
                                             "best_of", "start", "first_start", "status", "first", "first_key",
                                             "second", "second_key", "sets", "sets_won", "game", "serve",
                                             "winner", "ended")}
        self.events = {k: self.events[k] for k in keep}
        out = {"updated": iso(now), "live": [keep.get(d["k"], d) for d in self.live], "events": keep,
               "live_every_s": LIVE_EVERY}
        if self.errors:
            out["errors"] = self.errors[-5:]
        self.errors = []
        return out


class Publisher:
    """Keeps live.json as the only file of a one-commit 'live' branch (force-pushed)."""
    def __init__(self, root, branch="live"):
        self.dir, self.branch = os.path.join(root, ".live-branch"), branch
        run = lambda *a: subprocess.run(["git", "-C", root, *a], capture_output=True, text=True)
        run("worktree", "remove", "--force", self.dir)
        r = run("worktree", "add", "--detach", self.dir)
        if r.returncode:
            raise RuntimeError(r.stderr)
        self.git("checkout", "--orphan", "live-publish")
        self.git("rm", "-rf", "--quiet", ".")
        self.made = False

    def git(self, *a):
        return subprocess.run(["git", "-C", self.dir, *a], capture_output=True, text=True)

    def push(self, data):
        with open(os.path.join(self.dir, "live.json"), "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, separators=(",", ":"))
        self.git("add", "live.json")
        msg = f"live {data['updated']}"
        self.git("commit", "-q", "--amend", "-m", msg) if self.made else self.git("commit", "-q", "-m", msg)
        self.made = True
        r = self.git("push", "-q", "-f", "origin", f"HEAD:refs/heads/{self.branch}")
        if r.returncode:
            print(f"  · push failed: {r.stderr.strip()[:200]}", file=sys.stderr)


def previous(path, root):
    """The last published live.json (to keep first-seen start times across restarts)."""
    if os.path.exists(path):
        return u.load_json(path)
    subprocess.run(["git", "-C", root, "fetch", "-q", "--depth=1", "origin", "live"], capture_output=True)
    r = subprocess.run(["git", "-C", root, "show", "FETCH_HEAD:live.json"], capture_output=True, text=True)
    try:
        return json.loads(r.stdout) if r.returncode == 0 else None
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=345, help="stop after this long")
    ap.add_argument("--out", default="live.json")
    ap.add_argument("--no-publish", action="store_true", help="write --out only, don't push")
    args = ap.parse_args()
    key = os.environ.get("TENNIS_API_KEY", "").strip()
    if not key:
        sys.exit("TENNIS_API_KEY not set")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lv = Live(key, previous(args.out, root))
    pub = None if args.no_publish else Publisher(root)
    stop = time.time() + args.minutes * 60
    next_fx = 0.0
    while True:
        now = datetime.now(timezone.utc)
        if time.time() >= next_fx:
            lv.fixtures(now)
            next_fx = time.time() + FIXTURES_EVERY
        if lv.livescore(now):
            next_fx = 0.0                               # someone finished: confirm it next minute
        data = lv.snapshot(now)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, separators=(",", ":"))
        if pub:
            pub.push(data)
        print(f"{data['updated']}: {len(data['live'])} live, {len(data['events'])} events"
              + (f", errors {data['errors']}" if data.get("errors") else ""), file=sys.stderr, flush=True)
        if time.time() + LIVE_EVERY > stop:
            break
        time.sleep(LIVE_EVERY)


if __name__ == "__main__":
    main()
