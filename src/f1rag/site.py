"""Bundle an eval run into site/data/bundle.json for the static showcase. No API calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from f1rag.evaluate import summary

SYSTEMS = ("vector", "graph", "hybrid")


def build(run_dir: Path, out: Path, heldout_dir: Path | None = None) -> Path:
    bundle = _bundle(run_dir)
    if heldout_dir and heldout_dir.exists():
        extra = _bundle(heldout_dir)
        bundle["questions"] += extra["questions"]
        for system, stats in extra["summary"].items():
            main = bundle["summary"][system]
            n1, n2 = main["all"]["n"], stats["all"]["n"]
            for key in ("mean_tool_calls", "mean_seconds"):  # weighted over both sets
                main[key] = round((main[key] * n1 + stats[key] * n2) / (n1 + n2), 2)
            main["cost_usd"] = round(main["cost_usd"] + stats["cost_usd"], 4)
            if "held-out" in stats:
                main["held-out"] = stats["held-out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
    return out


def _bundle(run_dir: Path) -> dict:
    questions = json.loads((run_dir / "questions.json").read_text(encoding="utf-8"))
    results = {}
    for s in SYSTEMS:
        for f in (run_dir / s).glob("*.json"):
            r = json.loads(f.read_text(encoding="utf-8"))
            results.setdefault(r["qid"], {})[s] = {
                "answer": r["answer"],
                "correct": r["correct"],
                "trace": r["trace"],
                "cost_usd": round(r["cost_usd"], 5),
                "seconds": round(r["seconds"], 1),
                "tokens": r["input_tokens"] + r["output_tokens"],
                "error": r["error"],
            }
    model = next((json.loads(f.read_text(encoding="utf-8"))["model"] for f in (run_dir / "graph").glob("*.json")), "")
    bundle = {
        "model": model,
        "summary": summary(run_dir),
        "questions": [
            {k: q[k] for k in ("qid", "template", "hops", "question", "answer_type", "gold_text")} | {"results": results.get(q["qid"], {})}
            for q in questions
            if q["qid"] in results
        ],
    }
    return bundle


if __name__ == "__main__":
    args = sys.argv[1:]
    print(build(Path(args[0]), Path(args[1]), Path(args[2]) if len(args) > 2 else None))
