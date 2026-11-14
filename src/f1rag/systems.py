"""Three ways to answer an F1 question, all on the same Claude model:

* VectorRAG:  embed race and season documents, retrieve the top k, answer in one call.
* GraphAgent: Claude walks the knowledge graph through typed tools (manual tool-use loop).
* Hybrid:     the graph tools plus a vector_search tool.

Each returns the final answer plus a trace (what was retrieved or which tools were called), which the
showcase site replays.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from f1rag.data import F1Data
from f1rag.graph import GraphTools

MODEL = os.environ.get("F1RAG_MODEL", "claude-haiku-5-5")
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}
MAX_TURNS = 16  # the graph agent never needed more than 8; the vector agent sometimes did

ANSWER_RULES = (
    "Answer from the provided information only, not from memory. End with one line in the form "
    "'ANSWER: <answer>': a number, or names separated by commas (full names), or 'none'. "
    "If the information isn't enough, still give your best answer on the ANSWER line."
)


def price_per_mtok(model: str) -> tuple[float, float]:
    """(input, output) USD per million tokens. Unknown models fail loudly instead of costing $0."""
    if model not in PRICES:
        raise ValueError(f"no price for {model!r}; add it to PRICES in systems.py")
    return PRICES[model]


@dataclass
class Result:
    system: str
    answer: str
    raw: str
    trace: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    error: str | None = None
    model: str = MODEL

    @property
    def cost_usd(self) -> float:
        pin, pout = price_per_mtok(self.model)
        return (self.input_tokens * pin + self.output_tokens * pout) / 1e6


def extract_answer(text: str) -> str:
    """The text after the last 'ANSWER:', or the last non-empty line if the model forgot the marker."""
    hits = re.findall(r"ANSWER:\s*(.+)", text)
    if hits:
        return hits[-1].strip().rstrip(".")
    lines = text.strip().splitlines()
    return lines[-1] if lines else ""


def _text(content) -> str:
    return "\n".join(b.text for b in content if getattr(b, "type", "") == "text")


# ------------------------------------------------------------------ documents for vector search
def build_documents(d: F1Data, tools: GraphTools) -> list[dict]:
    docs = []
    for r in d.races:
        res = tools.race_result(r["season"], str(r["round"]), top=10)
        lines = [f"{x['position']}. {x['driver']} ({x['team']}), started {x['grid']}, {x['status']}" for x in res["results"]]
        docs.append(
            {
                "id": f"race:{r['race_id']}",
                "title": res["race"],
                "text": f"{res['race']}, round {r['round']} of the {r['season']} season, at {res['circuit']} on {r['date']}. "
                "Top 10: " + "; ".join(lines),
            }
        )
    for s in sorted({r["season"] for r in d.races}):
        ds = sorted((x for x in d.driver_standings if x["season"] == s), key=lambda x: x["position"] or 99)
        cs = sorted((x for x in d.constructor_standings if x["season"] == s), key=lambda x: x["position"] or 99)
        drivers = "; ".join(
            f"{x['position']}. {d.driver_name(x['driver'])} ({', '.join(d.constructor_name(c) for c in x['constructors'])}) "
            f"{x['points']:g} pts, {x['wins']} wins"
            for x in ds
        )
        teams = "; ".join(f"{x['position']}. {d.constructor_name(x['constructor'])} {x['points']:g} pts, {x['wins']} wins" for x in cs)
        docs.append(
            {
                "id": f"season:{s}",
                "title": f"{s} season standings",
                "text": f"{s} Formula 1 season final standings. Drivers: {drivers}. Constructors: {teams}.",
            }
        )
    return docs


class VectorIndex:
    def __init__(self, docs: list[dict], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.docs = docs
        self.encoder = SentenceTransformer(model_name)
        self.matrix = self.encoder.encode([d["text"] for d in docs], normalize_embeddings=True, show_progress_bar=False)

    def search(self, query: str, k: int = 8) -> list[tuple[dict, float]]:
        q = self.encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = self.matrix @ q
        top = np.argsort(-scores)[:k]
        return [(self.docs[i], float(scores[i])) for i in top]


# ------------------------------------------------------------------ systems
class VectorRAG:
    name = "vector"

    def __init__(self, client, index: VectorIndex, k: int = 8):
        self.client, self.index, self.k = client, index, k

    def answer(self, question: str) -> Result:
        t0 = time.monotonic()
        hits = self.index.search(question, self.k)
        context = "\n\n".join(f"[{i + 1}] {doc['text']}" for i, (doc, _) in enumerate(hits))
        resp = self.client.messages.create(
            model=MODEL,
            max_tokens=4000,
            system="You answer Formula 1 questions using retrieved documents. " + ANSWER_RULES,
            messages=[{"role": "user", "content": f"Documents:\n{context}\n\nQuestion: {question}"}],
        )
        raw = _text(resp.content)
        return Result(
            self.name,
            extract_answer(raw),
            raw,
            trace=[{"retrieved": [{"title": doc["title"], "score": round(s, 3)} for doc, s in hits]}],
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            seconds=time.monotonic() - t0,
            error="refusal" if resp.stop_reason == "refusal" else None,
        )


def _schema(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


GRAPH_TOOLS = [
    {
        "name": "find_entity",
        "description": "Resolve a driver, constructor (team) or circuit name to its exact entity. Returns candidates when ambiguous.",
        "input_schema": _schema(
            {"name": {"type": "string"}, "kind": {"type": "string", "enum": ["driver", "constructor", "circuit"]}}, ["name", "kind"]
        ),
    },
    {
        "name": "driver_career",
        "description": "A driver's seasons 2000-2025: teams, races, wins, podiums, championship position per season.",
        "input_schema": _schema({"driver": {"type": "string"}}, ["driver"]),
    },
    {
        "name": "teammates",
        "description": "A driver's team(s) and teammates in one season.",
        "input_schema": _schema({"driver": {"type": "string"}, "season": {"type": "integer"}}, ["driver", "season"]),
    },
    {
        "name": "driver_wins",
        "description": "Every Grand Prix a driver won (2000-2025), with the team.",
        "input_schema": _schema({"driver": {"type": "string"}}, ["driver"]),
    },
    {
        "name": "season_summary",
        "description": "A season's champions, number of races and top-5 driver and constructor standings.",
        "input_schema": _schema({"season": {"type": "integer"}}, ["season"]),
    },
    {
        "name": "race_result",
        "description": "Top-10 result of one race. race is a round number or part of the race/circuit/city name.",
        "input_schema": _schema({"season": {"type": "integer"}, "race": {"type": "string"}}, ["season", "race"]),
    },
    {
        "name": "team_season",
        "description": "A constructor's drivers, wins, points and championship position in one season.",
        "input_schema": _schema({"constructor": {"type": "string"}, "season": {"type": "integer"}}, ["constructor", "season"]),
    },
    {
        "name": "circuit_history",
        "description": "Every race held at a circuit (2000-2025) and its winner.",
        "input_schema": _schema({"circuit": {"type": "string"}}, ["circuit"]),
    },
    {"name": "champions", "description": "Drivers' and constructors' champion of every season 2000-2025.", "input_schema": _schema({}, [])},
]
for _t in GRAPH_TOOLS:
    _t["strict"] = True


def _vector_tool(description: str) -> dict:
    return {
        "name": "vector_search",
        "description": description,
        "input_schema": _schema({"query": {"type": "string"}}, ["query"]),
        "strict": True,
    }


# As a secondary tool for the hybrid agent (4 documents per call) and as the only tool of the vector agent.
VECTOR_TOOL = _vector_tool("Semantic search over race reports and season standings. Use for questions the graph tools don't cover.")
VECTOR_AGENT_TOOL = _vector_tool(
    "Semantic search over race reports (top 10 finishers per race) and season standings. "
    "Returns the 8 most similar documents. Call it as many times as needed with focused queries."
)

GRAPH_SYSTEM = (
    "You answer Formula 1 questions (2000-2025 seasons) by querying a knowledge graph with tools. "
    "Plan the chain of lookups, resolve ambiguous names with find_entity, and compute counts from the "
    "tool results rather than guessing. " + ANSWER_RULES
)
VECTOR_SYSTEM = (
    "You answer Formula 1 questions (2000-2025 seasons) by searching race reports and season standings "
    "with the vector_search tool. Search as many times as you need, with focused queries (for example one "
    "per season or race), and compute counts from what you retrieve rather than guessing. " + ANSWER_RULES
)


class ToolLoopAgent:
    """Claude in a manual tool-use loop over a fixed set of tools.

    The three agent systems differ only in name, system prompt, tool list and documents per search; the
    loop, the dispatch and the trace are shared, so their comparison measures the tools, not the code.
    """

    name: str
    system: str
    search_k = 4  # documents per vector_search call

    def __init__(self, client, tool_defs: list[dict], tools: GraphTools | None = None, index: VectorIndex | None = None):
        self.client, self.tool_defs, self.tools, self.index = client, tool_defs, tools, index

    def _handlers(self) -> dict[str, Callable[[dict], dict]]:
        """An explicit name -> function table, limited to the tools this agent was given.

        The model can only reach what's listed here; it can't call arbitrary methods by name.
        """
        t = self.tools
        table: dict[str, Callable[[dict], dict]] = {
            "find_entity": lambda a: t.find(a["name"], a["kind"]),
            "driver_career": lambda a: t.driver_career(a["driver"]),
            "teammates": lambda a: t.teammates(a["driver"], a["season"]),
            "driver_wins": lambda a: t.driver_wins(a["driver"]),
            "season_summary": lambda a: t.season_summary(a["season"]),
            "race_result": lambda a: t.race_result(a["season"], a["race"]),
            "team_season": lambda a: t.team_season(a["constructor"], a["season"]),
            "circuit_history": lambda a: t.circuit_history(a["circuit"]),
            "champions": lambda a: t.champions(),
            "vector_search": self._vector_search,
        }
        allowed = {tool["name"] for tool in self.tool_defs}
        return {name: fn for name, fn in table.items() if name in allowed}

    def _vector_search(self, args: dict) -> dict:
        hits = self.index.search(args["query"], self.search_k)
        return {"results": [{"title": d["title"], "text": d["text"]} for d, _ in hits]}

    def _run_tool(self, name: str, args: dict) -> dict:
        handler = self._handlers().get(name)
        if handler is None:
            return {"error": f"unknown tool {name!r}"}
        try:
            return handler(args)
        except LookupError as e:
            return {"error": e.args[0]}
        except Exception as e:  # noqa: BLE001 - a bad tool call is reported to the model, not raised
            return {"error": f"{type(e).__name__}: {e}"}

    def answer(self, question: str) -> Result:
        t0 = time.monotonic()
        messages = [{"role": "user", "content": question}]
        result = Result(self.name, "", "")
        for _ in range(MAX_TURNS):
            resp = self.client.messages.create(model=MODEL, max_tokens=8000, system=self.system, tools=self.tool_defs, messages=messages)
            result.input_tokens += resp.usage.input_tokens
            result.output_tokens += resp.usage.output_tokens
            if resp.stop_reason == "refusal":
                result.error = "refusal"
                break
            if resp.stop_reason == "pause_turn":
                messages.append({"role": "assistant", "content": resp.content})
                continue
            calls = [b for b in resp.content if b.type == "tool_use"]
            if not calls:
                result.raw = _text(resp.content)
                break
            messages.append({"role": "assistant", "content": resp.content})
            messages.append({"role": "user", "content": [self._call(c, result) for c in calls]})
        else:
            result.error = "max turns reached"
        result.answer = extract_answer(result.raw) if result.raw else ""
        result.seconds = time.monotonic() - t0
        return result

    def _call(self, call, result: Result) -> dict:
        """Run one tool call, record it in the trace, and build its tool_result block."""
        out = self._run_tool(call.name, call.input)
        text = json.dumps(out, ensure_ascii=False)
        result.trace.append({"tool": call.name, "input": call.input, "output": text[:1500]})
        block = {"type": "tool_result", "tool_use_id": call.id, "content": text}
        if "error" in out:
            block["is_error"] = True
        return block


class GraphAgent(ToolLoopAgent):
    """Walks the knowledge graph through the typed tools."""

    name = "graph"
    system = GRAPH_SYSTEM

    def __init__(self, client, tools: GraphTools):
        super().__init__(client, GRAPH_TOOLS, tools=tools)


class HybridAgent(ToolLoopAgent):
    """The graph tools plus vector search for anything they don't cover."""

    name = "hybrid"
    system = GRAPH_SYSTEM

    def __init__(self, client, tools: GraphTools, index: VectorIndex):
        super().__init__(client, GRAPH_TOOLS + [VECTOR_TOOL], tools=tools, index=index)


class VectorAgent(ToolLoopAgent):
    """A stronger vector baseline: the same loop, with vector search as its only tool.

    It can issue several searches with its own queries (for example one per season) and combine the
    results, which plain top-k retrieval can't.
    """

    name = "vector-agent"
    system = VECTOR_SYSTEM
    search_k = 8  # the same k as plain vector RAG, so the only difference is the search loop

    def __init__(self, client, index: VectorIndex):
        super().__init__(client, [VECTOR_AGENT_TOOL], index=index)
