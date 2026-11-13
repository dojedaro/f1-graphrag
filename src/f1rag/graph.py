"""The knowledge graph and the typed tools an agent uses to walk it.

Nodes (keyed ``(kind, id)``): Driver, Constructor, Season, Race, Circuit.
Edges (MultiDiGraph keys):
    (Driver)-[RACED {position, position_text, points, grid, status, constructor}]->(Race)
    (Driver)-[DROVE_FOR:<season>]->(Constructor)
    (Race)-[HELD_AT]->(Circuit)
    (Race)-[IN_SEASON]->(Season)
    (Season)-[STANDING {position, points, wins}]->(Driver | Constructor)   final championship tables
    (Season)-[CHAMPION]->(Driver), (Season)-[CONSTRUCTORS_CHAMPION]->(Constructor)

Every tool answers by walking these edges; none reads the source tables. The tools are the design
decision that matters: each answers one question an analyst would ask, returns names rather than ids,
and reports ambiguity instead of guessing.
"""

from __future__ import annotations

import difflib
import unicodedata
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

import networkx as nx

from f1rag.data import F1Data

Node = tuple[str, Any]
UNCLASSIFIED = 99  # sort key for championship entries without a position


def norm(text: str) -> str:
    """Lowercase, strip accents and punctuation, so 'Räikkönen' matches 'raikkonen'."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return " ".join("".join(ch if ch.isalnum() else " " for ch in text.lower()).split())


def build_graph(d: F1Data) -> nx.MultiDiGraph:
    """Build the graph in data order; tools rely on edge insertion order for stable output."""
    g = nx.MultiDiGraph()
    for k, v in d.drivers.items():
        g.add_node(("driver", k), kind="Driver", **v)
    for k, v in d.constructors.items():
        g.add_node(("constructor", k), kind="Constructor", **v)
    for k, v in d.circuits.items():
        g.add_node(("circuit", k), kind="Circuit", **v)
    for r in d.races:
        race = ("race", r["race_id"])
        g.add_node(race, kind="Race", name=r["name"], season=r["season"], round=r["round"], date=r["date"])
        g.add_node(("season", r["season"]), kind="Season", name=str(r["season"]))
        g.add_edge(race, ("circuit", r["circuit"]), key="HELD_AT")
        g.add_edge(race, ("season", r["season"]), key="IN_SEASON")
    for row in d.results:
        driver = ("driver", row["driver"])
        g.add_edge(
            driver,
            ("race", row["race_id"]),
            key="RACED",
            position=row["position"],
            position_text=row["position_text"],
            points=row["points"],
            grid=row["grid"],
            status=row["status"],
            constructor=row["constructor"],
        )
        g.add_edge(driver, ("constructor", row["constructor"]), key=f"DROVE_FOR:{row['season']}", season=row["season"])
    for kind, rows in (("driver", d.driver_standings), ("constructor", d.constructor_standings)):
        for s in rows:
            target = (kind, s[kind])
            g.add_edge(("season", s["season"]), target, key="STANDING", position=s["position"], points=s["points"], wins=s["wins"])
            if s["position"] == 1:
                g.add_edge(("season", s["season"]), target, key="CHAMPION" if kind == "driver" else "CONSTRUCTORS_CHAMPION")
    return g


class GraphTools:
    """Typed queries that walk the graph. Every method returns plain data an LLM can read."""

    def __init__(self, g: nx.MultiDiGraph):
        self.g = g
        self._names: dict[str, dict[str, str]] = defaultdict(dict)
        for (kind, key), attrs in g.nodes(data=True):
            if kind in ("driver", "constructor", "circuit"):
                self._names[kind][key] = attrs["name"]

    @classmethod
    def from_data(cls, d: F1Data) -> GraphTools:
        return cls(build_graph(d))

    # ---- graph helpers ------------------------------------------------------------------------
    def _out(self, node: Node, key: str) -> Iterator[tuple[Node, dict]]:
        """Targets of a node's outgoing edges with this key (or key prefix ending in ':')."""
        for _, target, k, attrs in self.g.out_edges(node, keys=True, data=True):
            if k == key or (key.endswith(":") and k.startswith(key)):
                yield target, attrs

    def _in(self, node: Node, key: str) -> Iterator[tuple[Node, dict]]:
        for source, _, k, attrs in self.g.in_edges(node, keys=True, data=True):
            if k == key:
                yield source, attrs

    def _name(self, node: Node) -> str:
        return self.g.nodes[node]["name"]

    def _race_label(self, race: Node) -> str:
        r = self.g.nodes[race]
        return f"{r['season']} {r['name']}"

    def _standings(self, season: int, kind: str) -> list[tuple[Node, dict]]:
        rows = [(n, a) for n, a in self._out(("season", season), "STANDING") if n[0] == kind]
        return sorted(rows, key=lambda row: row[1]["position"] or UNCLASSIFIED)

    def _seasons(self) -> list[int]:
        return sorted(key for kind, key in self.g.nodes if kind == "season")

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

    def _resolve(self, kind: str, value: str) -> Node:
        """An exact id, or a name that resolves to exactly one entity; LookupError otherwise."""
        if value in self._names[kind]:
            return (kind, value)
        hit = self.find(value, kind)
        if "id" not in hit:
            raise LookupError(hit)
        return (kind, hit["id"])

    # ---- driver-centric -----------------------------------------------------------------------
    def driver_career(self, driver: str) -> dict:
        """Teams per season, with wins, podiums and championship position."""
        node = self._resolve("driver", driver)
        by_season: dict[int, list[dict]] = defaultdict(list)
        for race, attrs in self._out(node, "RACED"):
            by_season[self.g.nodes[race]["season"]].append(attrs)
        position = {s[1]: a["position"] for s, a in self._in(node, "STANDING")}
        seasons = [
            {
                "season": s,
                "teams": sorted({self._name(("constructor", r["constructor"])) for r in rows}),
                "races": len(rows),
                "wins": sum(r["position"] == 1 for r in rows),
                "podiums": sum(r["position"] <= 3 for r in rows),
                "championship_position": position.get(s),
            }
            for s, rows in sorted(by_season.items())
        ]
        return {"driver": self._name(node), "seasons": seasons}

    def teammates(self, driver: str, season: int) -> dict:
        """Driver -> teams that season (DROVE_FOR) -> other drivers with a DROVE_FOR edge to them that season."""
        node = self._resolve("driver", driver)
        key = f"DROVE_FOR:{season}"
        teams = {team for team, _ in self._out(node, key)}
        mates = {other for team in teams for other, _ in self._in(team, key) if other != node}
        return {
            "driver": self._name(node),
            "season": season,
            "teams": sorted(self._name(t) for t in teams),
            "teammates": sorted(self._name(m) for m in mates),
        }

    def driver_wins(self, driver: str) -> dict:
        node = self._resolve("driver", driver)
        wins = [(race, a) for race, a in self._out(node, "RACED") if a["position"] == 1]
        return {
            "driver": self._name(node),
            "total": len(wins),
            "wins": [{"race": self._race_label(race), "team": self._name(("constructor", a["constructor"]))} for race, a in wins],
        }

    # ---- season / race-centric ----------------------------------------------------------------
    def season_summary(self, season: int) -> dict:
        drivers, teams = self._standings(season, "driver"), self._standings(season, "constructor")
        return {
            "season": season,
            "races": sum(1 for _ in self._in(("season", season), "IN_SEASON")),
            "drivers_champion": self._name(drivers[0][0]) if drivers else None,
            "constructors_champion": self._name(teams[0][0]) if teams else None,
            "top_drivers": [
                {"position": a["position"], "driver": self._name(n), "points": a["points"], "wins": a["wins"]} for n, a in drivers[:5]
            ],
            "top_constructors": [
                {"position": a["position"], "team": self._name(n), "points": a["points"], "wins": a["wins"]} for n, a in teams[:5]
            ],
        }

    def race_result(self, season: int, race: str, top: int = 10) -> dict:
        """race can be a round number or part of the race, circuit or city name ('Monaco')."""
        races = [r for r, _ in self._in(("season", season), "IN_SEASON")]

        def circuit(r: Node) -> dict:
            return self.g.nodes[next(c for c, _ in self._out(r, "HELD_AT"))]

        if str(race).isdigit():
            match = [r for r in races if self.g.nodes[r]["round"] == int(race)]
        else:
            q = norm(str(race))
            match = [
                r for r in races if q in norm(self.g.nodes[r]["name"]) or q in norm(circuit(r)["name"]) or q in norm(circuit(r)["locality"])
            ]
        if len(match) != 1:
            return {
                "error": f"{len(match)} races in {season} match {race!r}",
                "races": [f"round {self.g.nodes[r]['round']}: {self.g.nodes[r]['name']}" for r in (match or races)],
            }
        (r,) = match
        rows = sorted(self._in(r, "RACED"), key=lambda row: row[1]["position"])
        info = self.g.nodes[r]
        return {
            "race": self._race_label(r),
            "round": info["round"],
            "circuit": circuit(r)["name"],
            "date": info["date"],
            "results": [
                {
                    "position": a["position_text"],
                    "driver": self._name(drv),
                    "team": self._name(("constructor", a["constructor"])),
                    "grid": a["grid"],
                    "status": a["status"],
                }
                for drv, a in rows[:top]
            ],
        }

    def team_season(self, constructor: str, season: int) -> dict:
        node = self._resolve("constructor", constructor)
        drivers = [drv for drv, _ in self._in(node, f"DROVE_FOR:{season}")]
        wins = sum(
            1
            for drv in drivers
            for race, a in self._out(drv, "RACED")
            if a["position"] == 1 and a["constructor"] == node[1] and self.g.nodes[race]["season"] == season
        )
        standing = (
            next((a for n, a in self._out(("season", season), "STANDING") if n == node), None) if ("season", season) in self.g else None
        )
        return {
            "team": self._name(node),
            "season": season,
            "drivers": sorted(self._name(drv) for drv in drivers),
            "wins": wins,
            "championship_position": standing["position"] if standing else None,
            "points": standing["points"] if standing else None,
        }

    def circuit_history(self, circuit: str) -> dict:
        hit = self.find(circuit, "circuit")
        if "id" not in hit:
            return hit
        races = []
        for race, _ in self._in(("circuit", hit["id"]), "HELD_AT"):
            winner = next((drv for drv, a in self._in(race, "RACED") if a["position"] == 1), None)
            info = self.g.nodes[race]
            races.append({"season": info["season"], "race": info["name"], "winner": self._name(winner) if winner else None})
        return {"circuit": hit["name"], "races": races}

    def champions(self) -> dict:
        """Drivers' and constructors' champion for every season in the data."""
        seasons = []
        for s in self._seasons():
            summary = self.season_summary(s)
            seasons.append(
                {"season": s, "drivers_champion": summary["drivers_champion"], "constructors_champion": summary["constructors_champion"]}
            )
        return {"seasons": seasons}
