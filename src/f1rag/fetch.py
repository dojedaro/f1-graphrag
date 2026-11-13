"""Download Formula 1 results from the Jolpica API (the open successor to Ergast) and cache them.

Every response is saved under data/raw/, so the network is only touched once per page. The API allows
4 requests per second and 500 per hour without a token; this client sends at most one per second.
Data: Jolpica F1 (https://github.com/jolpica/jolpica-f1), free for non-commercial use.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

BASE = "https://api.jolpi.ca/ergast/f1"
USER_AGENT = "f1-graphrag (portfolio project)"
PAGE = 100
MIN_INTERVAL_S = 1.1

_last = 0.0


def _get(url: str) -> dict:
    global _last
    wait = MIN_INTERVAL_S - (time.monotonic() - _last)
    if wait > 0:
        time.sleep(wait)
    _last = time.monotonic()
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 4:
                time.sleep(10 * (attempt + 1))
                continue
            raise
    raise RuntimeError("unreachable")


def cached(raw_dir: Path, name: str, url: str) -> dict:
    path = raw_dir / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = _get(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def season_results(raw_dir: Path, season: int) -> list[dict]:
    """All race results of a season, one dict per race (with its Results list)."""
    races: dict[str, dict] = {}
    offset = 0
    while True:
        data = cached(raw_dir, f"{season}/results_{offset}", f"{BASE}/{season}/results/?limit={PAGE}&offset={offset}")
        table = data["MRData"]
        for race in table["RaceTable"]["Races"]:
            # A race spanning two pages arrives in two parts; merge them by round.
            entry = races.setdefault(race["round"], {**race, "Results": []})
            entry["Results"].extend(race["Results"])
        offset += PAGE
        if offset >= int(table["total"]):
            break
    return [races[k] for k in sorted(races, key=int)]


def season_standings(raw_dir: Path, season: int, kind: str) -> list[dict]:
    """Final driver or constructor standings. kind is 'driver' or 'constructor'."""
    data = cached(raw_dir, f"{season}/{kind}standings", f"{BASE}/{season}/{kind}standings/?limit={PAGE}")
    lists = data["MRData"]["StandingsTable"]["StandingsLists"]
    return lists[0][f"{kind.capitalize()}Standings"] if lists else []


def fetch_all(raw_dir: Path, first: int, last: int, log=print) -> None:
    for season in range(first, last + 1):
        races = season_results(raw_dir, season)
        season_standings(raw_dir, season, "driver")
        season_standings(raw_dir, season, "constructor")
        log(f"{season}: {len(races)} races, {sum(len(r['Results']) for r in races)} results")


if __name__ == "__main__":
    import sys

    fetch_all(Path("data/raw"), int(sys.argv[1]), int(sys.argv[2]))
