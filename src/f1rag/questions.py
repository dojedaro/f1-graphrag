"""Generate questions with gold answers computed from the data, and score answers against them.

Every gold answer is computed by code from the same tables the systems read, so the benchmark needs no
hand labeling and no LLM judge. Questions are tagged by how many facts must be chained to answer them.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass

from f1rag.data import F1Data
from f1rag.graph import GraphTools, norm


@dataclass
class Question:
    qid: str
    template: str
    hops: str  # "1-hop", "2-hop", "3-hop" or "aggregation"
    question: str
    answer_type: str  # "names" or "number"
    gold: list[str] | int  # driver/constructor ids, or a number
    gold_text: str
    entity_kind: str = "driver"

    def to_dict(self) -> dict:
        return asdict(self)


class Generator:
    def __init__(self, d: F1Data, tools: GraphTools, seed: int = 7):
        self.d, self.t, self.rng = d, tools, random.Random(seed)
        self.winner = {r["race_id"]: r for r in d.results if r["position"] == 1}
        self.teams_of = defaultdict(set)  # (driver, season) -> constructors
        for r in d.results:
            self.teams_of[(r["driver"], r["season"])].add(r["constructor"])
        self.champion = {s["season"]: s["driver"] for s in d.driver_standings if s["position"] == 1}
        self.con_champion = {s["season"]: s["constructor"] for s in d.constructor_standings if s["position"] == 1}
        self.seasons = sorted({r["season"] for r in d.races})

    def _names(self, ids, kind="driver") -> str:
        f = self.d.driver_name if kind == "driver" else self.d.constructor_name
        return ", ".join(sorted(f(i) for i in ids))

    # ---------------------------------------------------------------- 1-hop
    def race_winner(self):
        race = self.rng.choice(self.d.races)
        w = self.winner[race["race_id"]]["driver"]
        return ("1-hop", f"Who won the {race['season']} {race['name']}?", "names", [w], self._names([w]), "driver")

    def drivers_champion(self):
        s = self.rng.choice(self.seasons[:-1] or self.seasons)
        c = self.champion[s]
        return ("1-hop", f"Who won the {s} Formula 1 drivers' championship?", "names", [c], self._names([c]), "driver")

    def driver_team(self):
        (drv, s), teams = self.rng.choice([(k, v) for k, v in self.teams_of.items() if len(v) == 1])
        (t,) = teams
        return (
            "1-hop",
            f"Which team did {self.d.driver_name(drv)} drive for in {s}?",
            "names",
            [t],
            self._names([t], "constructor"),
            "constructor",
        )

    # ---------------------------------------------------------------- 2-hop
    def teammate(self):
        while True:
            (drv, s), teams = self.rng.choice([(k, v) for k, v in self.teams_of.items() if len(v) == 1])
            mates = {r["driver"] for r in self.d.results if r["season"] == s and r["constructor"] in teams and r["driver"] != drv}
            if len(mates) == 1:
                return (
                    "2-hop",
                    f"Who was {self.d.driver_name(drv)}'s teammate for the whole {s} season?",
                    "names",
                    sorted(mates),
                    self._names(mates),
                    "driver",
                )

    def champion_wins(self):
        s = self.rng.choice(self.seasons)
        c = self.champion[s]
        n = sum(1 for r in self.d.results if r["season"] == s and r["driver"] == c and r["position"] == 1)
        return ("2-hop", f"How many races did the {s} drivers' champion win that season?", "number", n, str(n), "driver")

    def constructors_champion_at_first_win(self):
        firsts = {}
        for r in sorted(self.d.results, key=lambda r: (r["season"], r["round"])):
            if r["position"] == 1 and r["driver"] not in firsts:
                firsts[r["driver"]] = r["season"]
        drv = self.rng.choice(sorted(firsts))
        s = firsts[drv]
        c = self.con_champion[s]
        return (
            "2-hop",
            f"Which team won the constructors' championship in the season {self.d.driver_name(drv)} "
            "won a Grand Prix for the first time (counting from 2000)?",
            "names",
            [c],
            self._names([c], "constructor"),
            "constructor",
        )

    # ---------------------------------------------------------------- 3-hop
    def champion_teammates(self):
        champs = set(self.champion.values())
        pool = []
        for drv in sorted({k[0] for k in self.teams_of}):
            mates = set()
            for (dd, s), teams in self.teams_of.items():
                if dd == drv:
                    mates |= {r["driver"] for r in self.d.results if r["season"] == s and r["constructor"] in teams and r["driver"] != drv}
            hit = sorted(mates & champs)
            if hit:
                pool.append((drv, hit))
        drv, hit = self.rng.choice(pool)
        return (
            "3-hop",
            f"Which of {self.d.driver_name(drv)}'s teammates (2000-2025) won a drivers' championship in 2000-2025?",
            "names",
            hit,
            self._names(hit),
            "driver",
        )

    def winner_team_next_season(self):
        race = self.rng.choice([r for r in self.d.races if r["season"] < self.seasons[-1]])
        team = self.winner[race["race_id"]]["constructor"]
        n = sum(1 for r in self.d.results if r["season"] == race["season"] + 1 and r["constructor"] == team and r["position"] == 1)
        return (
            "3-hop",
            f"How many races did the team that won the {race['season']} {race['name']} win in the following season?",
            "number",
            n,
            str(n),
            "driver",
        )

    # ---------------------------------------------------------------- aggregation
    def team_count(self):
        drv = self.rng.choice(sorted({k[0] for k in self.teams_of}))
        n = len({t for (dd, _), ts in self.teams_of.items() if dd == drv for t in ts})
        return (
            "aggregation",
            f"For how many different teams did {self.d.driver_name(drv)} start a race between 2000 and 2025?",
            "number",
            n,
            str(n),
            "driver",
        )

    def wins_with_team(self):
        pairs = defaultdict(int)
        for r in self.winner.values():
            pairs[(r["driver"], r["constructor"])] += 1
        (drv, team), n = self.rng.choice(sorted(pairs.items()))
        return (
            "aggregation",
            f"How many Grands Prix did {self.d.driver_name(drv)} win driving for {self.d.constructor_name(team)} between 2000 and 2025?",
            "number",
            n,
            str(n),
            "driver",
        )

    # ---------------------------------------------------------------- held-out
    # Written after the graph tools were designed, so no single tool answers them: the agent has to
    # compose lookups. These test the tool design, not just the tools.
    def distinct_winners(self):
        s = self.rng.choice(self.seasons)
        n = len({w["driver"] for w in self.winner.values() if w["season"] == s})
        return ("held-out", f"How many different drivers won at least one race in the {s} season?", "number", n, str(n), "driver")

    def pole_to_win(self):
        s = self.rng.choice(self.seasons)
        n = sum(1 for w in self.winner.values() if w["season"] == s and w["grid"] == 1)
        return (
            "held-out",
            f"In how many races of the {s} season did the driver who started from pole position (grid 1) win?",
            "number",
            n,
            str(n),
            "driver",
        )

    def circuit_most_wins(self):
        pool = []
        for cid in sorted(self.d.circuits):
            counts = defaultdict(int)
            for r in self.d.races:
                if r["circuit"] == cid and r["race_id"] in self.winner:
                    counts[self.winner[r["race_id"]]["driver"]] += 1
            if sum(counts.values()) >= 5:
                top = max(counts.values())
                leaders = [k for k, v in counts.items() if v == top]
                if len(leaders) == 1:
                    pool.append((cid, leaders[0]))
        cid, drv = self.rng.choice(pool)
        return (
            "held-out",
            f"Which driver won the most races at {self.d.circuits[cid]['name']} between 2000 and 2025?",
            "names",
            [drv],
            self._names([drv]),
            "driver",
        )

    def podiums_without_win(self):
        pool = sorted({(r["driver"], r["season"]) for r in self.d.results if r["position"] in (2, 3)})
        drv, s = self.rng.choice(pool)
        n = sum(1 for r in self.d.results if r["driver"] == drv and r["season"] == s and r["position"] in (2, 3))
        return (
            "held-out",
            f"How many times did {self.d.driver_name(drv)} finish second or third in a race in {s}?",
            "number",
            n,
            str(n),
            "driver",
        )

    def runner_up_team(self):
        pool = [x for x in self.d.driver_standings if x["position"] == 2 and len(x["constructors"]) == 1]
        x = self.rng.choice(pool)
        t = x["constructors"][0]
        return (
            "held-out",
            f"Which team did the runner-up of the {x['season']} drivers' championship drive for?",
            "names",
            [t],
            self._names([t], "constructor"),
            "constructor",
        )

    HELDOUT = ("distinct_winners", "pole_to_win", "circuit_most_wins", "podiums_without_win", "runner_up_team")

    TEMPLATES = (
        "race_winner",
        "drivers_champion",
        "driver_team",
        "teammate",
        "champion_wins",
        "constructors_champion_at_first_win",
        "champion_teammates",
        "winner_team_next_season",
        "team_count",
        "wins_with_team",
    )

    def generate(self, per_template: int, templates: tuple[str, ...] | None = None) -> list[Question]:
        out, seen = [], set()
        for name in templates or self.TEMPLATES:
            made = 0
            while made < per_template:
                hops, text, atype, gold, gold_text, kind = getattr(self, name)()
                if text in seen:
                    continue
                seen.add(text)
                made += 1
                out.append(Question(f"{name}-{made:02d}", name, hops, text, atype, gold, gold_text, kind))
        return out


# ---------------------------------------------------------------- scoring
_NUMBER_WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}


def parse_number(text: str) -> int | None:
    m = re.search(r"-?\d+", text)
    if m:
        return int(m.group())
    for w in norm(text).split():
        if w in _NUMBER_WORDS:
            return _NUMBER_WORDS[w]
    return None


def parse_names(text: str, tools: GraphTools, kind: str) -> set[str] | None:
    """Resolve 'Lewis Hamilton and Alonso' to ids. Returns None if any part can't be resolved."""
    if norm(text) in {"none", "nobody", "no one"}:
        return set()
    parts = [p for p in re.split(r",|;|\band\b|&|/", text) if p.strip()]
    ids = set()
    for p in parts:
        hit = tools.find(p.strip(), kind)
        if "id" not in hit:
            return None
        ids.add(hit["id"])
    return ids


def score(q: Question, answer: str, tools: GraphTools) -> bool:
    if q.answer_type == "number":
        return parse_number(answer) == q.gold
    got = parse_names(answer, tools, q.entity_kind)
    return got is not None and got == set(q.gold)
