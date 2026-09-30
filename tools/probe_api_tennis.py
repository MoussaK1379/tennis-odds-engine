"""One-off probe: fetch small samples from api-tennis.com so the importer can be
built against real responses. The API key is removed from everything saved.

    TENNIS_API_KEY=... python tools/probe_api_tennis.py --out samples/api_tennis
"""
import argparse, json, os, sys, urllib.parse, urllib.request
from collections import Counter
from datetime import date, timedelta

BASE = "https://api.api-tennis.com/tennis/"


def call(key, method, **params):
    q = urllib.parse.urlencode({"method": method, "APIkey": key, **params})
    try:
        with urllib.request.urlopen(f"{BASE}?{q}", timeout=90) as r:
            return r.status, json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        return e.code, {"http_error": e.code, "body": e.read().decode("utf-8", "replace")[:2000]}
    except Exception as e:                                   # noqa: BLE001
        return None, {"error": repr(e)}


def trim_event(ev):
    ev = dict(ev)
    pbp = ev.get("pointbypoint")
    if isinstance(pbp, list):
        ev["pointbypoint"] = pbp[:2] + ([f"... {len(pbp) - 2} more games"] if len(pbp) > 2 else [])
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="samples/api_tennis")
    args = ap.parse_args()
    key = os.environ.get("TENNIS_API_KEY", "").strip()
    if not key:
        sys.exit("TENNIS_API_KEY not set")
    os.makedirs(args.out, exist_ok=True)

    def save(name, obj):
        text = json.dumps(obj, ensure_ascii=False, indent=1).replace(key, "***")
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as fh:
            fh.write(text)

    report = {}
    today = date.today()

    st, ev_types = call(key, "get_events")
    report["get_events"] = {"http": st, "success": ev_types.get("success"), "error": ev_types.get("error")}
    save("get_events.json", ev_types)

    windows = {"past": (today - timedelta(days=3), today - timedelta(days=1)),
               "upcoming": (today, today + timedelta(days=1))}
    for label, (a, b) in windows.items():
        st, fx = call(key, "get_fixtures", date_start=a.isoformat(), date_stop=b.isoformat())
        res = fx.get("result") if isinstance(fx, dict) else None
        info = {"http": st, "success": fx.get("success"), "error": fx.get("error") or fx.get("http_error"),
                "window": [a.isoformat(), b.isoformat()]}
        if isinstance(res, list):
            info["count"] = len(res)
            info["event_type_type"] = Counter(e.get("event_type_type") for e in res).most_common()
            info["event_status"] = Counter(e.get("event_status") for e in res).most_common(15)
            info["keys"] = sorted({k for e in res for k in e})
            info["with_statistics"] = sum(1 for e in res if e.get("statistics"))
            info["with_pointbypoint"] = sum(1 for e in res if e.get("pointbypoint"))
            picks = []
            for tour in ("Atp Singles", "Wta Singles"):
                done = [e for e in res if e.get("event_type_type") == tour]
                done.sort(key=lambda e: (not e.get("statistics"), not e.get("pointbypoint")))
                picks += [trim_event(e) for e in done[:3]]
            save(f"fixtures_{label}_sample.json", picks)
            stat_names = Counter()
            for e in res:
                for s in e.get("statistics") or []:
                    stat_names[(s.get("stat_period"), s.get("stat_type"), s.get("stat_name"))] += 1
            info["statistics_names"] = [list(k) + [n] for k, n in stat_names.most_common(60)]
        else:
            save(f"fixtures_{label}_raw.json", fx)
        report[f"get_fixtures_{label}"] = info

    save("report.json", report)
    print(json.dumps(report, indent=1)[:4000].replace(key, "***"))


if __name__ == "__main__":
    main()
