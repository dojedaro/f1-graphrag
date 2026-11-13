"""Run every system on every question, save each result as JSON, and summarize by hop count.

Results are written one file per (system, question), so an interrupted run resumes where it stopped.
Usage: python -m f1rag.evaluate run [--per-template N] [--out runs/NAME]
       python -m f1rag.evaluate summary runs/NAME
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from f1rag.data import load
from f1rag.graph import GraphTools, build_graph
from f1rag.questions import Generator, Question, score


def load_env(path: Path = Path(".env")) -> None:
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            k, sep, v = line.strip().partition("=")
            if sep and k and not k.startswith("#"):
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def run(per_template: int, out: Path, heldout: bool = False, only: list[str] | None = None) -> None:
    import anthropic

    from f1rag.systems import MODEL, GraphAgent, HybridAgent, VectorAgent, VectorIndex, VectorRAG, build_documents

    d = load(Path("data/raw"))
    tools = GraphTools(d, build_graph(d))
    gen = Generator(d, tools, seed=11 if heldout else 7)
    questions = gen.generate(per_template, gen.HELDOUT if heldout else None)
    out.mkdir(parents=True, exist_ok=True)
    (out / "questions.json").write_text(json.dumps([q.to_dict() for q in questions], indent=1, ensure_ascii=False), encoding="utf-8")
    index = VectorIndex(build_documents(d, tools))
    client = anthropic.Anthropic()
    systems = [VectorRAG(client, index), VectorAgent(client, tools, index), GraphAgent(client, tools), HybridAgent(client, tools, index)]
    if only:
        systems = [s for s in systems if s.name in only]
    for q in questions:
        for s in systems:
            path = out / s.name / f"{q.qid}.json"
            if path.exists():
                continue
            try:
                r = s.answer(q.question)
            except Exception as e:  # noqa: BLE001 - one failed call shouldn't end the run
                print(f"[{s.name}] {q.qid}: FAILED {type(e).__name__}: {e}"[:200], flush=True)
                continue
            rec = {**asdict(r), "qid": q.qid, "model": MODEL, "cost_usd": r.cost_usd, "correct": score(q, r.answer, tools)}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"[{s.name:6}] {q.qid}: {'ok ' if rec['correct'] else 'BAD'} {r.answer[:50]!r} (gold {q.gold_text[:40]!r})", flush=True)


def summary(out: Path) -> dict:
    questions = {q["qid"]: Question(**q) for q in json.loads((out / "questions.json").read_text(encoding="utf-8"))}
    rows: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    cost, calls, secs = defaultdict(float), defaultdict(int), defaultdict(float)
    models: dict = defaultdict(set)
    for sys_dir in sorted(p for p in out.iterdir() if p.is_dir()):
        for f in sys_dir.glob("*.json"):
            rec = json.loads(f.read_text(encoding="utf-8"))
            q = questions[rec["qid"]]
            for key in (q.hops, "all"):
                rows[sys_dir.name][key][0] += rec["correct"]
                rows[sys_dir.name][key][1] += 1
            cost[sys_dir.name] += rec["cost_usd"]
            calls[sys_dir.name] += sum(1 for t in rec["trace"] if "tool" in t)
            secs[sys_dir.name] += rec["seconds"]
            models[sys_dir.name].add(rec.get("model", ""))
    result = {}
    for system, by in rows.items():
        n = by["all"][1]
        result[system] = {
            **{k: {"correct": c, "n": t, "accuracy": round(c / t, 3), "ci": wilson(c, t)} for k, (c, t) in by.items()},
            "n_missing": len(questions) - n,  # failed calls write nothing; never let them shrink the denominator silently
            "cost_usd": round(cost[system], 4),
            "models": sorted(models[system]),  # more than one means costs and results mix models
            "mean_tool_calls": round(calls[system] / n, 2),
            "mean_seconds": round(secs[system] / n, 1),
        }
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--per-template", type=int, default=12)
    r.add_argument("--out", type=Path, default=Path("runs/eval"))
    r.add_argument("--heldout", action="store_true", help="the question types written after the tools")
    r.add_argument("--systems", help="comma-separated subset: vector,vector-agent,graph,hybrid")
    s = sub.add_parser("summary")
    s.add_argument("out", type=Path)
    args = p.parse_args()
    load_env()
    if args.cmd == "run":
        run(args.per_template, args.out, args.heldout, args.systems.split(",") if args.systems else None)
    res = summary(args.out)
    mixed = {k: v["models"] for k, v in res.items() if len(v["models"]) > 1}
    if mixed:
        print(f"WARNING: results from more than one model in a system: {mixed}")
    missing = {k: v["n_missing"] for k, v in res.items() if v["n_missing"]}
    if missing:
        print(f"WARNING: questions with no saved result: {missing}. Re-run to fill them before reporting.")
    order = ["1-hop", "2-hop", "3-hop", "aggregation", "held-out", "all"]
    print(f"\n{'system':8}" + "".join(f"{k:>14}" for k in order) + f"{'cost':>9}{'tools/q':>9}{'s/q':>7}")
    for system, v in res.items():
        cells = "".join(f"{v[k]['correct']:>6}/{v[k]['n']:<3}{v[k]['accuracy']:>5.0%}" if k in v else f"{'-':>14}" for k in order)
        print(f"{system:8}{cells}{v['cost_usd']:>9.3f}{v['mean_tool_calls']:>9}{v['mean_seconds']:>7}")
    (args.out / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
