"""Summary and site-bundle arithmetic on tiny synthetic runs (no API calls)."""

import json
from pathlib import Path

from f1rag.evaluate import summary
from f1rag.site import build


def make_run(root: Path, hops: str, records: dict[str, list[tuple[bool, int, float, float]]], n_questions: int) -> Path:
    """records: system -> list of (correct, tool_calls, seconds, cost)."""
    qs = [
        {
            "qid": f"q{i}",
            "template": "t",
            "hops": hops,
            "question": f"Q{i}?",
            "answer_type": "number",
            "gold": 1,
            "gold_text": "1",
            "entity_kind": "driver",
        }
        for i in range(n_questions)
    ]
    (root / "questions.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "questions.json").write_text(json.dumps(qs), encoding="utf-8")
    for system, rows in records.items():
        for i, (ok, calls, secs, cost) in enumerate(rows):
            rec = {
                "system": system,
                "qid": f"q{i}",
                "answer": "1",
                "raw": "",
                "correct": ok,
                "model": "m",
                "trace": [{"tool": "x"}] * calls,
                "input_tokens": 1,
                "output_tokens": 1,
                "seconds": secs,
                "cost_usd": cost,
                "error": None,
            }
            path = root / system / f"q{i}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rec), encoding="utf-8")
    return root


def test_summary_counts_and_missing(tmp_path):
    run = make_run(tmp_path / "a", "1-hop", {"graph": [(True, 2, 1.0, 0.1), (False, 4, 3.0, 0.3)]}, n_questions=3)
    s = summary(run)["graph"]
    assert s["all"] == {"correct": 1, "n": 2, "accuracy": 0.5, "ci": s["all"]["ci"]}
    assert s["n_missing"] == 1  # one question has no saved result
    assert s["mean_tool_calls"] == 3.0 and s["mean_seconds"] == 2.0 and s["cost_usd"] == 0.4


def test_bundle_merges_heldout_with_weighted_means(tmp_path):
    systems = ("vector", "vector-agent", "graph", "hybrid")
    main = make_run(tmp_path / "main", "1-hop", {s: [(True, 2, 1.0, 0.1)] * 3 for s in systems}, n_questions=3)
    held = make_run(tmp_path / "held", "held-out", {s: [(True, 8, 5.0, 0.2)] for s in systems}, n_questions=1)
    out = build(main, tmp_path / "bundle.json", held)
    g = json.loads(out.read_text(encoding="utf-8"))["summary"]["graph"]
    assert g["held-out"]["n"] == 1 and g["all"]["n"] == 3
    assert g["mean_tool_calls"] == round((2 * 3 + 8 * 1) / 4, 2)
    assert g["cost_usd"] == round(0.3 + 0.2, 4)
