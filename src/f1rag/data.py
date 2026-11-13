"""Normalize cached Jolpica responses into plain tables (lists of dicts) that everything else reads."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class F1Data:
    drivers: dict[str, dict] = field(default_factory=dict)  # driverId -> {name, nationality, dob}
    constructors: dict[str, dict] = field(default_factory=dict)  # constructorId -> {name, nationality}
    circuits: dict[str, dict] = field(default_factory=dict)  # circuitId -> {name, locality, country}
    races: list[dict] = field(default_factory=list)  # {race_id, season, round, name, circuit, date}
    results: list[dict] = field(default_factory=list)  # one row per car per race
    driver_standings: list[dict] = field(default_factory=list)  # final standings per season
    constructor_standings: list[dict] = field(default_factory=list)

    def driver_name(self, driver_id: str) -> str:
        return self.drivers[driver_id]["name"]

    def constructor_name(self, constructor_id: str) -> str:
        return self.constructors[constructor_id]["name"]

    def race(self, race_id: str) -> dict:
        return self._race_index[race_id]

    def index(self) -> F1Data:
        self._race_index = {r["race_id"]: r for r in self.races}
        return self


def load(raw_dir: Path) -> F1Data:
    d = F1Data()
    for season_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        season = int(season_dir.name)
        races: dict[int, dict] = {}
        for page in sorted(season_dir.glob("results_*.json"), key=lambda p: int(p.stem.split("_")[1])):
            for race in json.loads(page.read_text(encoding="utf-8"))["MRData"]["RaceTable"]["Races"]:
                rnd = int(race["round"])
                c = race["Circuit"]
                d.circuits[c["circuitId"]] = {
                    "name": c["circuitName"],
                    "locality": c["Location"]["locality"],
                    "country": c["Location"]["country"],
                }
                race_id = f"{season}-{rnd:02d}"
                races.setdefault(
                    rnd,
                    {
                        "race_id": race_id,
                        "season": season,
                        "round": rnd,
                        "name": race["raceName"],
                        "circuit": c["circuitId"],
                        "date": race["date"],
                    },
                )
                for r in race["Results"]:
                    drv, con = r["Driver"], r["Constructor"]
                    d.drivers[drv["driverId"]] = {
                        "name": f"{drv['givenName']} {drv['familyName']}",
                        "nationality": drv.get("nationality", ""),
                        "dob": drv.get("dateOfBirth", ""),
                    }
                    d.constructors[con["constructorId"]] = {"name": con["name"], "nationality": con.get("nationality", "")}
                    d.results.append(
                        {
                            "race_id": race_id,
                            "season": season,
                            "round": rnd,
                            "driver": drv["driverId"],
                            "constructor": con["constructorId"],
                            "grid": int(r.get("grid", 0) or 0),
                            "position": int(r["position"]),
                            "position_text": r["positionText"],
                            "points": float(r["points"]),
                            "status": r["status"],
                        }
                    )
        d.races.extend(races[k] for k in sorted(races))
        for kind, target in (("driver", d.driver_standings), ("constructor", d.constructor_standings)):
            path = season_dir / f"{kind}standings.json"
            lists = json.loads(path.read_text(encoding="utf-8"))["MRData"]["StandingsTable"]["StandingsLists"]
            if not lists:
                continue
            for s in lists[0][f"{kind.capitalize()}Standings"]:
                row = {
                    "season": season,
                    "position": int(s.get("position", 0) or 0),
                    "points": float(s["points"]),
                    "wins": int(s["wins"]),
                }
                if kind == "driver":
                    row["driver"] = s["Driver"]["driverId"]
                    row["constructors"] = [c["constructorId"] for c in s["Constructors"]]
                else:
                    row["constructor"] = s["Constructor"]["constructorId"]
                target.append(row)
    return d.index()
