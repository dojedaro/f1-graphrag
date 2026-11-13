"""Offline tests: the graph, the tools, the question generator and the scorer. No API calls."""

from pathlib import Path

import pytest

from f1rag.data import load
from f1rag.graph import GraphTools, norm
from f1rag.questions import Generator, parse_names, parse_number, score
from f1rag.systems import extract_answer

DATA = load(Path(__file__).parents[1] / "data" / "raw")
TOOLS = GraphTools.from_data(DATA)


def test_known_facts():
    s = TOOLS.season_summary(2010)
    assert (s["drivers_champion"], s["constructors_champion"]) == ("Sebastian Vettel", "Red Bull")
    assert TOOLS.teammates("Lewis Hamilton", 2007)["teammates"] == ["Fernando Alonso"]
    assert TOOLS.race_result(2024, "Monaco", 1)["results"][0]["driver"] == "Charles Leclerc"


def test_entity_resolution_handles_accents_and_ambiguity():
    assert norm("Räikkönen") == "raikkonen"
    assert TOOLS.find("raikkonen")["name"] == "Kimi Räikkönen"
    assert len(TOOLS.find("Schumacher")["ambiguous"]) == 3
    assert "error" in TOOLS.find("Ayrton Senna")  # before 2000: not in the data


def test_ambiguous_tool_input_is_an_error_not_a_guess():
    with pytest.raises(LookupError):
        TOOLS.driver_career("Schumacher")


def test_every_gold_answer_passes_its_own_scorer():
    qs = Generator(DATA, TOOLS, seed=7).generate(12)
    assert len(qs) == 120 and len({q.question for q in qs}) == 120
    assert all(score(q, q.gold_text, TOOLS) for q in qs)


def test_generation_is_deterministic():
    a = [q.question for q in Generator(DATA, TOOLS, seed=7).generate(3)]
    b = [q.question for q in Generator(DATA, TOOLS, seed=7).generate(3)]
    assert a == b


def test_answer_parsing():
    assert extract_answer("thinking...\nANSWER: Mark Webber.") == "Mark Webber"
    assert parse_number("They won twelve races") == 12 and parse_number("ANSWER 7") == 7
    assert parse_names("Hamilton and Fernando Alonso", TOOLS, "driver") == {"hamilton", "alonso"}
    assert parse_names("Schumacher", TOOLS, "driver") is None  # ambiguous answers don't score
    assert parse_names("none", TOOLS, "driver") == set()


def test_tools_answer_from_the_graph_alone():
    """No tables involved: a hand-built graph is enough, so the tools really traverse edges."""
    import networkx as nx

    g = nx.MultiDiGraph()
    for key, name in [("a", "Ann Able"), ("b", "Bo Bright"), ("c", "Cy Clear")]:
        g.add_node(("driver", key), name=name)
    for key, name in [("red", "Red Team"), ("blue", "Blue Team")]:
        g.add_node(("constructor", key), name=name)
    g.add_node(("season", 2030), name="2030")
    g.add_edge(("driver", "a"), ("constructor", "red"), key="DROVE_FOR:2030", season=2030)
    g.add_edge(("driver", "b"), ("constructor", "red"), key="DROVE_FOR:2030", season=2030)
    g.add_edge(("driver", "c"), ("constructor", "blue"), key="DROVE_FOR:2030", season=2030)

    result = GraphTools(g).teammates("Ann Able", 2030)
    assert result["teams"] == ["Red Team"] and result["teammates"] == ["Bo Bright"]
