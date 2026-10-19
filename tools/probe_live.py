"""One-off probe of api-tennis's live data, for building the Live tab: saves a trimmed
sample of get_livescore and of today's fixtures (statuses, score fields, times) to
data/probe/, with the API key removed. Run by .github/workflows/probe.yml.

    TENNIS_API_KEY=... python tools/probe_live.py
"""
import json, os, sys
from collections import Counter
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import update_data as u                           # noqa: E402

OUT = "data/probe"
TYPES = {"Atp Singles", "Wta Singles", "Challenger Men Singles", "Challenger Women Singles"}


def trim(ev):
    ev = dict(ev)
    pbp = ev.get("pointbypoint")
    if isinstance(pbp, list):
        ev["pointbypoint"] = pbp[-2:] + [f"... {len(pbp)} games in all"]
    ev.pop("statistics", None)
    return ev


def main():
    key = os.environ["TENNIS_API_KEY"].strip()
    os.makedirs(OUT, exist_ok=True)
    rep = {}
    def save(name, obj):
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, ensure_ascii=False, indent=1, default=str).replace(key, "***"))
    try:
        live = u.api_tennis_call(key, method="get_livescore")
    except Exception as e:                           # noqa: BLE001
        live, rep["livescore_error"] = [], str(e).replace(key, "***")
    live = live if isinstance(live, list) else []
    rep["livescore_count"] = len(live)
    rep["livescore_types"] = Counter(e.get("event_type_type") for e in live).most_common()
    rep["livescore_status"] = Counter(e.get("event_status") for e in live).most_common()
    rep["livescore_keys"] = sorted({k for e in live for k in e})
    ours = [e for e in live if e.get("event_type_type") in TYPES]
    save("livescore_sample.json", [trim(e) for e in ours[:6]])
    today = date.today()
    fx = []
    for t, (tk, tn) in u.API_TENNIS_TYPES.items():
        try:
            got = u.api_tennis_call(key, method="get_fixtures", event_type_key=tk, timezone="Etc/UTC",
                                    date_start=today.isoformat(), date_stop=(today + timedelta(days=1)).isoformat())
        except Exception as e:                       # noqa: BLE001
            rep[f"fixtures_{t}_error"] = str(e).replace(key, "***")
            continue
        fx += [e for e in got if e.get("event_type_type") == tn]
    rep["fixtures_status"] = Counter(e.get("event_status") for e in fx).most_common()
    rep["fixtures_live_flag"] = Counter(e.get("event_live") for e in fx).most_common()
    rep["fixture_keys"] = sorted({k for e in fx for k in e})
    by = {}
    for e in fx:
        by.setdefault(e.get("event_status") or "(blank)", []).append(trim(e))
    save("fixtures_by_status.json", {k: v[:3] for k, v in by.items()})
    save("report.json", rep)
    print(json.dumps(rep, indent=1)[:3000])


if __name__ == "__main__":
    main()
