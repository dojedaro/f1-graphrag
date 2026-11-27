"""Draw the README chart from the saved run summaries: accuracy by question type for each system.

python scripts/plot_results.py            # writes assets/accuracy_by_type.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = [Path("runs/eval-2026-11-13/summary.json"), Path("runs/heldout-2026-11-13/summary.json")]
SYSTEMS = [
    ("vector", "Vector RAG (one shot)", "#9aa5b1"),
    ("vector-agent", "Vector agent (search loop)", "#5b8def"),
    ("graph", "Graph agent (typed tools)", "#1f9d55"),
]
GROUPS = [
    ("1-hop", "1-hop"),
    ("2-hop", "2-hop"),
    ("3-hop", "3-hop"),
    ("aggregation", "Counting"),
    ("held-out", "Held-out"),
    ("all", "All 160"),
]


def load() -> dict[str, dict[str, tuple[int, int]]]:
    out: dict[str, dict[str, tuple[int, int]]] = {}
    for path in RUNS:
        for system, groups in json.loads(path.read_text(encoding="utf-8")).items():
            for group, v in groups.items():
                if isinstance(v, dict) and "correct" in v:
                    c, n = out.setdefault(system, {}).get(group, (0, 0))
                    out[system][group] = (c + v["correct"], n + v["n"])
    return out


def main() -> None:
    data = load()
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    width = 0.27
    for i, (system, label, color) in enumerate(SYSTEMS):
        xs, ys = [], []
        for j, (group, _) in enumerate(GROUPS):
            c, n = data[system][group]
            xs.append(j + (i - 1) * width)
            ys.append(100 * c / n)
        bars = ax.bar(xs, ys, width, label=label, color=color)
        ax.bar_label(bars, labels=[f"{y:.0f}" for y in ys], fontsize=7, padding=2)
    ax.set_xticks(range(len(GROUPS)), [g[1] for g in GROUPS])
    ax.set_ylim(0, 112)
    ax.set_ylabel("Correct (%)")
    ax.set_title("160 F1 questions, same model (Claude Haiku 5.5), exact scoring", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=3, bbox_to_anchor=(0, -0.28))
    fig.tight_layout()
    out = Path("assets/accuracy_by_type.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
