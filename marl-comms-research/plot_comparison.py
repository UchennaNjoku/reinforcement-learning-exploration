"""Comparison figure and summary table across all conditions and maps.

Reads eval JSON files and produces:
  1. A 3-panel grouped bar chart (capture rate, avg steps, collision rate)
  2. A printed markdown summary table

Usage:
    python plot_comparison.py
    python plot_comparison.py --out results/comparison.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Data registry — canonical eval results (seed 99, 200 episodes)
# ---------------------------------------------------------------------------

RESULTS = {
    # (condition, map) → {capture_rate, avg_steps, collision_rate}
    ("Random",   "easy_open"):    {"capture_rate": 0.00,  "avg_steps": 300.0,  "collision_rate": 0.049},
    ("Random",   "center_block"): {"capture_rate": 0.63,  "avg_steps": 207.71, "collision_rate": 0.066},
    ("Random",   "split_barrier"):{"capture_rate": 0.535, "avg_steps": 234.83, "collision_rate": 0.066},

    ("No-Comm",  "easy_open"):    {"capture_rate": 1.00,  "avg_steps": 14.49,  "collision_rate": 0.001},
    ("No-Comm",  "center_block"): {"capture_rate": 0.995, "avg_steps": 40.52,  "collision_rate": 0.014},
    ("No-Comm",  "split_barrier"):{"capture_rate": 0.990, "avg_steps": 60.02,  "collision_rate": 0.155},

    ("Comm-4",   "easy_open"):    {"capture_rate": 1.00,  "avg_steps": 9.47,   "collision_rate": 0.0002},
    ("Comm-4",   "center_block"): {"capture_rate": 1.00,  "avg_steps": 22.91,  "collision_rate": 0.301},
    ("Comm-4",   "split_barrier"):{"capture_rate": 0.820, "avg_steps": 152.7,  "collision_rate": 0.281},

    ("Comm-16",  "easy_open"):    {"capture_rate": 1.00,  "avg_steps": 11.13,  "collision_rate": 0.008},
    ("Comm-16",  "center_block"): {"capture_rate": 1.00,  "avg_steps": 27.41,  "collision_rate": 0.019},
    ("Comm-16",  "split_barrier"):{"capture_rate": 0.965, "avg_steps": 113.23, "collision_rate": 0.289},
}

CONDITIONS = ["Random", "No-Comm", "Comm-4", "Comm-16"]
MAPS       = ["easy_open", "center_block", "split_barrier"]
MAP_LABELS = ["easy_open", "center_block", "split_barrier"]

COLORS = {
    "Random":  "#aec7e8",
    "No-Comm": "#ff9896",
    "Comm-4":  "#98df8a",
    "Comm-16": "#ffbb78",
}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(out_path: Path) -> None:
    n_maps   = len(MAPS)
    n_conds  = len(CONDITIONS)
    x        = np.arange(n_maps)
    width    = 0.18
    offsets  = np.linspace(-(n_conds - 1) / 2, (n_conds - 1) / 2, n_conds) * width

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "Pursuit Performance: No-Comm vs Communication Conditions\n"
        "(trained on easy_open, seed 0 — evaluated seed 99, 200 episodes)",
        fontsize=12, fontweight="bold",
    )

    metrics = [
        ("capture_rate",   "Capture Rate (%)",   True,  (0, 115)),
        ("avg_steps",      "Avg Steps to End",   False, None),
        ("collision_rate", "Collision Rate (%)", True,  (0, 42)),
    ]

    for ax, (key, label, as_pct, ylim) in zip(axes, metrics):
        for i, cond in enumerate(CONDITIONS):
            vals = [RESULTS[(cond, m)][key] for m in MAPS]
            if as_pct:
                vals = [v * 100 for v in vals]
            bars = ax.bar(
                x + offsets[i], vals, width,
                label=cond, color=COLORS[cond], edgecolor="white", linewidth=0.5,
            )
            for bar, v in zip(bars, vals):
                fmt = f"{v:.0f}" if as_pct else f"{v:.0f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    fmt, ha="center", va="bottom", fontsize=6,
                )

        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(MAP_LABELS, fontsize=8, rotation=10)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    legend_patches = [
        mpatches.Patch(color=COLORS[c], label=c) for c in CONDITIONS
    ]
    axes[0].legend(handles=legend_patches, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Comparison figure saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------

def print_table() -> None:
    header = f"{'Condition':<10} {'Map':<15} {'Cap%':>6} {'Steps':>8} {'Coll%':>7}"
    print("\n" + header)
    print("-" * len(header))
    for cond in CONDITIONS:
        for m in MAPS:
            r = RESULTS[(cond, m)]
            print(
                f"{cond:<10} {m:<15} "
                f"{r['capture_rate']*100:>5.1f}% "
                f"{r['avg_steps']:>8.1f} "
                f"{r['collision_rate']*100:>6.1f}%"
            )
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate comparison figure and table")
    p.add_argument("--out", default="results/comparison.png")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print_table()
    plot(out_path)
