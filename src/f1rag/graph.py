"""The knowledge graph and the typed tools an agent uses to walk it.

Nodes: Driver, Constructor, Season, Race, Circuit.
Edges: (Driver)-[RACED {position, points, grid, status}]->(Race),
       (Driver)-[DROVE_FOR {season}]->(Constructor), (Race)-[HELD_AT]->(Circuit),
       (Race)-[IN_SEASON]->(Season), (Season)-[CHAMPION]->(Driver),
       (Season)-[CONSTRUCTORS_CHAMPION]->(Constructor).

The tools are the design decision that matters. Each answers one question a human analyst would ask,
returns compact JSON-ready data with names (never bare ids), and tells the caller when an entity is
ambiguous instead of guessing.
"""

from __future__ import annotations

import difflib
import unicodedata
from collections import defaultdict

import networkx as nx

from f1rag.data import F1Data


def norm(text: str) -> str:
    """Lowercase, strip accents and punctuation, so 'Räikkönen' matches 'raikkonen'."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return " ".join("".join(ch if ch.isalnum() else " " for ch in text.lower()).split())


def build_graph(d: F1Data) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for k, v in d.drivers.items():
        g.add_node(("driver", k), kind="Driver", **v)
    for k, v in d.constructors.items():
        g.add_node(("constructor", k), kind="Constructor", **v)
    for k, v in d.circuits.items():
        g.add_node(("circuit", k), kind="Circuit", **v)
    for r in d.races:
        rid = ("race", r["race_id"])
        g.add_node(rid, kind="Race", name=f"{r['season']} {r['name']}", season=r["season"], round=r["round"], date=r["date"])
        g.add_node(("season", r["season"]), kind="Season", name=str(r["season"]))
        g.add_edge(rid, ("circuit", r["circuit"]), key="HELD_AT")
        g.add_edge(rid, ("season", r["season"]), key="IN_SEASON")
    seen_team = set()
    for row in d.results:
        g.add_edge(
            ("driver", row["driver"]),
            ("race", row["race_id"]),
            key="RACED",
            position=row["position"],
            points=row["points"],
            grid=row["grid"],
            status=row["status"],
            constructor=row["constructor"],
        )
        k = (row["driver"], row["constructor"], row["season"])
        if k not in seen_team:
            seen_team.add(k)
            g.add_edge(
                ("driver", row["driver"]), ("constructor", row["constructor"]), key=f"DROVE_FOR:{row['season']}", season=row["season"]
            )
    for s in d.driver_standings:
        if s["position"] == 1:
            g.add_edge(("season", s["season"]), ("driver", s["driver"]), key="CHAMPION")
    for s in d.constructor_standings:
        if s["position"] == 1:
            g.add_edge(("season", s["season"]), ("constructor", s["constructor"]), key="CONSTRUCTORS_CHAMPION")
    return g


class GraphTools:
    """Typed queries over the graph. Every method returns plain data an LLM can read."""

    def __init__(self, d: F1Data, g: nx.MultiDiGraph):
        self.d, self.g = d, g
        self._names = {
            "driver": {k: v["name"] for k, v in d.drivers.items()},
            "constructor": {k: v["name"] for k, v in d.constructors.items()},
            "circuit": {k: v["name"] for k, v in d.circuits.items()},
        }
        self._by_driver_season: dict[tuple[str, int], list[dict]] = defaultdict(list)
        for row in d.results:
            self._by_driver_season[(row["driver"], row["season"])].append(row)

    # ---- entity resolution ------------------------------------------------------------------
    def find(self, name: str, kind: str = "driver") -> dict:
        """Resolve a name to one entity, or list the candidates when it's ambiguous."""
        q = norm(name)
        table = self._names[kind]
        exact = [k for k, v in table.items() if norm(v) == q or k == q.replace(" ", "_")]
        partial = exact or [k for k, v in table.items() if q and (q in norm(v) or norm(v).split()[-1] == q)]
        if not partial:
            close = difflib.get_close_matches(q, [norm(v) for v in table.values()], n=3, cutoff=0.75)
            partial = [k for k, v in table.items() if norm(v) in close]
        if len(partial) == 1:
            return {"id": partial[0], "name": table[partial[0]]}
        if not partial:
            return {"error": f"no {kind} matches {name!r}"}
        return {"ambiguous": [{"id": k, "name": table[k]} for k in partial[:8]]}

    def _driver(self, driver: str) -> str:
        if driver in self.d.drivers:
            return driver
        hit = self.find(driver, "driver")
        if "id" not in hit:
            raise LookupError(hit)
        return hit["id"]

    def _constructor(self, constructor: str) -> str:
        if constructor in self.d.constructors:
            return constructor
        hit = self.find(constructor, "constructor")
        if "id" not in hit:
            raise LookupError(hit)
        return hit["id"]

    # ---- driver-centric -----------------------------------------------------------------------
    def driver_career(self, driver: str) -> dict:
        """Teams per season, with wins, podiums and championship position."""
        did = self._driver(driver)
        standings = {s["season"]: s for s in self.d.driver_standings if s["driver"] == did}
        seasons = sorted({s for (dd, s) in self._by_driver_season if dd == did})
        out = []
        for s in seasons:
            rows = self._by_driver_season[(did, s)]
            teams = sorted({self.d.constructor_name(r["constructor"]) for r in rows})
            out.append(
                {
                    "season": s,
                    "teams": teams,
                    "races": len(rows),
                    "wins": sum(r["position"] == 1 for r in rows),
                    "podiums": sum(r["position"] <= 3 for r in rows),
                    "championship_position": standings.get(s, {}).get("position"),
                }
            )
        return {"driver": self.d.driver_name(did), "seasons": out}

    def teammates(self, driver: str, season: int) -> dict:
        did = self._driver(driver)
        teams = {r["constructor"] for r in self._by_driver_season[(did, season)]}
        mates = sorted(
            {
                self.d.driver_name(r["driver"])
                for r in self.d.results
                if r["season"] == season and r["constructor"] in teams and r["driver"] != did
            }
        )
        return {
            "driver": self.d.driver_name(did),
            "season": season,
            "teams": sorted(self.d.constructor_name(t) for t in teams),
            "teammates": mates,
        }

    def driver_wins(self, driver: str) -> dict:
        did = self._driver(driver)
        wins = [r for r in self.d.results if r["driver"] == did and r["position"] == 1]
        return {
            "driver": self.d.driver_name(did),
            "total": len(wins),
            "wins": [
                {"race": f"{w['season']} {self.d.race(w['race_id'])['name']}", "team": self.d.constructor_name(w["constructor"])}
                for w in wins
            ],
        }

    # ---- season / race-centric ----------------------------------------------------------------
    def season_summary(self, season: int) -> dict:
        ds = sorted((s for s in self.d.driver_standings if s["season"] == season), key=lambda s: s["position"] or 99)
        cs = sorted((s for s in self.d.constructor_standings if s["season"] == season), key=lambda s: s["position"] or 99)
        return {
            "season": season,
            "races": sum(r["season"] == season for r in self.d.races),
            "drivers_champion": self.d.driver_name(ds[0]["driver"]) if ds else None,
            "constructors_champion": self.d.constructor_name(cs[0]["constructor"]) if cs else None,
            "top_drivers": [
                {"position": s["position"], "driver": self.d.driver_name(s["driver"]), "points": s["points"], "wins": s["wins"]}
                for s in ds[:5]
            ],
            "top_constructors": [
                {"position": s["position"], "team": self.d.constructor_name(s["constructor"]), "points": s["points"], "wins": s["wins"]}
                for s in cs[:5]
            ],
        }

    def race_result(self, season: int, race: str, top: int = 10) -> dict:
        """race can be a round number or part of the race name ('Monaco')."""
        races = [r for r in self.d.races if r["season"] == season]
        if str(race).isdigit():
            match = [r for r in races if r["round"] == int(race)]
        else:
            q = norm(str(race))
            match = [
                r
                for r in races
                if q in norm(r["name"])
                or q in norm(self.d.circuits[r["circuit"]]["name"])
                or q in norm(self.d.circuits[r["circuit"]]["locality"])
            ]
        if len(match) != 1:
            return {
                "error": f"{len(match)} races in {season} match {race!r}",
                "races": [f"round {r['round']}: {r['name']}" for r in (match or races)],
            }
        r = match[0]
        rows = sorted((x for x in self.d.results if x["race_id"] == r["race_id"]), key=lambda x: x["position"])
        return {
            "race": f"{season} {r['name']}",
            "round": r["round"],
            "circuit": self.d.circuits[r["circuit"]]["name"],
            "date": r["date"],
            "results": [
                {
                    "position": x["position_text"],
                    "driver": self.d.driver_name(x["driver"]),
                    "team": self.d.constructor_name(x["constructor"]),
                    "grid": x["grid"],
                    "status": x["status"],
                }
                for x in rows[:top]
            ],
        }

    def team_season(self, constructor: str, season: int) -> dict:
        cid = self._constructor(constructor)
        rows = [r for r in self.d.results if r["constructor"] == cid and r["season"] == season]
        st = next((s for s in self.d.constructor_standings if s["constructor"] == cid and s["season"] == season), None)
        return {
            "team": self.d.constructor_name(cid),
            "season": season,
            "drivers": sorted({self.d.driver_name(r["driver"]) for r in rows}),
            "wins": sum(r["position"] == 1 for r in rows),
            "championship_position": st["position"] if st else None,
            "points": st["points"] if st else None,
        }

    def circuit_history(self, circuit: str) -> dict:
        hit = self.find(circuit, "circuit")
        if "id" not in hit:
            return hit
        races = [r for r in self.d.races if r["circuit"] == hit["id"]]
        winners = []
        for r in races:
            w = next((x for x in self.d.results if x["race_id"] == r["race_id"] and x["position"] == 1), None)
            winners.append({"season": r["season"], "race": r["name"], "winner": self.d.driver_name(w["driver"]) if w else None})
        return {"circuit": hit["name"], "races": winners}

    def champions(self) -> dict:
        """Drivers' and constructors' champion for every season in the data."""
        return {
            "seasons": [
                {
                    "season": s,
                    "drivers_champion": self.season_summary(s)["drivers_champion"],
                    "constructors_champion": self.season_summary(s)["constructors_champion"],
                }
                for s in sorted({r["season"] for r in self.d.races})
            ]
        }
