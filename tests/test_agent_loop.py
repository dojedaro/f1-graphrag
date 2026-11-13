"""The graph agent's tool loop, driven by a scripted fake client (no API calls)."""

from pathlib import Path
from types import SimpleNamespace as NS

from f1rag.data import load
from f1rag.graph import GraphTools, build_graph
from f1rag.systems import MAX_TURNS, GraphAgent

DATA = load(Path(__file__).parents[1] / "data" / "raw")
TOOLS = GraphTools(DATA, build_graph(DATA))


def tool_use(name, args, id_="t1"):
    return NS(type="tool_use", name=name, input=args, id=id_)


def text(t):
    return NS(type="text", text=t)


def reply(*blocks, stop="end_turn"):
    return NS(content=list(blocks), stop_reason=stop, usage=NS(input_tokens=100, output_tokens=10))


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.messages = NS(create=self._create)

    def _create(self, **kw):
        self.requests.append(kw)
        return self.replies.pop(0)


def test_tool_call_then_answer():
    client = FakeClient(
        [
            reply(tool_use("season_summary", {"season": 2010}), stop="tool_use"),
            reply(text("The champion was Vettel.\nANSWER: Sebastian Vettel")),
        ]
    )
    r = GraphAgent(client, TOOLS).answer("Who won the 2010 title?")
    assert r.answer == "Sebastian Vettel" and r.error is None
    assert [t["tool"] for t in r.trace] == ["season_summary"]
    assert "Sebastian Vettel" in r.trace[0]["output"]
    # The tool result went back in the next request, tied to the tool_use id.
    sent = client.requests[1]["messages"][-1]["content"][0]
    assert sent["type"] == "tool_result" and sent["tool_use_id"] == "t1"
    assert (r.input_tokens, r.output_tokens) == (200, 20)


def test_bad_tool_input_is_reported_to_the_model_not_raised():
    client = FakeClient(
        [
            reply(tool_use("driver_career", {"driver": "Schumacher"}), stop="tool_use"),
            reply(text("ANSWER: none")),
        ]
    )
    r = GraphAgent(client, TOOLS).answer("q")
    sent = client.requests[1]["messages"][-1]["content"][0]
    assert sent.get("is_error") is True and "ambiguous" in sent["content"]
    assert r.answer == "none"


def test_turn_limit_stops_a_looping_agent():
    client = FakeClient([reply(tool_use("champions", {}), stop="tool_use") for _ in range(MAX_TURNS)])
    r = GraphAgent(client, TOOLS).answer("q")
    assert r.error == "max turns reached" and len(client.requests) == MAX_TURNS


def test_refusal_is_recorded():
    r = GraphAgent(FakeClient([reply(stop="refusal")]), TOOLS).answer("q")
    assert r.error == "refusal" and r.answer == ""
