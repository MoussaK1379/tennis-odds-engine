"""Which played matches never appeared on the site's match list, and why that might be.

Every refresh commits matches.json, so its git history is a record of everything the list
ever showed. This compares that with the finished matches in the results cache
(data/api_tennis_matches.csv) for a date range, by tour / level and main draw / qualifying,
and lists the tournaments with the most misses. A clone needs the history: run
`git fetch --depth=600 origin main` first in a shallow clone.

    python tools/audit_list.py 2026-10-14 2026-10-20
"""

import collections, csv, json, os, subprocess, sys
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import update_matches as um                       # noqa: E402


def shown_pairs(ref="origin/main"):
    """{date: set of player-pair keys} over every committed matches.json, and the snapshot times."""
    log = subprocess.run(["git", "-C", ROOT, "log", ref, "--format=%H %cI", "--", "matches.json"],
                         capture_output=True, text=True).stdout.split("\n")
    shown, times = collections.defaultdict(set), []
    for line in filter(None, log):
        h, t = line.split()
        try:
            d = json.loads(subprocess.run(["git", "-C", ROOT, "show", f"{h}:matches.json"],
                                          capture_output=True, text=True).stdout)
        except ValueError:
            continue
        times.append(t)
        for m in d.get("matches", []) + d.get("unmatched", []):
            if m.get("start"):
                shown[m["start"][:10]].add(frozenset((um.loose_key(m["a"]["name"]), um.loose_key(m["b"]["name"]))))
    return shown, sorted(times)


def main():
    lo, hi = sys.argv[1], sys.argv[2]
    shown, times = shown_pairs()
    print(f"{len(times)} list snapshots, {times[0] if times else '-'} .. {times[-1] if times else '-'}")
    with open(os.path.join(ROOT, "data", "api_tennis_matches.csv"), encoding="utf-8") as fh:
        cache = list(csv.DictReader(fh))
    tot, miss, where = collections.Counter(), collections.Counter(), collections.defaultdict(collections.Counter)
    for r in cache:
        if not (lo <= r["date"] <= hi) or r["status"] == "wo":
            continue
        k = (r["tour"], "qualifying" if r["qualifying"] == "True" else "main draw")
        tot[k] += 1
        pair = frozenset((um.loose_key(r["winner"]), um.loose_key(r["loser"])))
        d = date.fromisoformat(r["date"])
        if not any(pair in shown.get((d + timedelta(days=j)).isoformat(), set()) for j in (-1, 0, 1)):
            miss[k] += 1
            where[k][(r["date"], r["tournament"])] += 1
    for k in sorted(tot):
        print(f"{k[0]:7} {k[1]:10} played {tot[k]:4}  never listed {miss[k]:4}  ({miss[k] / tot[k]:.0%})")
        for (d, t), n in where[k].most_common(5):
            print(f"          {d} {t}: {n}")


if __name__ == "__main__":
    main()
