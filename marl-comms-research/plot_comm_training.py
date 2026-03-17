"""Training curve comparison for all three conditions (baseline, comm4, comm16).

Reads training logs and produces a 3-panel figure overlaying all conditions:
  1. Capture rate (rolling average)
  2. Average steps per episode (rolling average)
  3. Epsilon decay

Usage:
    python plot_comm_training.py
    python plot_comm_training.py --out results/comm_training_curves.png
    python plot_comm_training.py --window 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


LOGS = [
    {
        "label":  "No-Comm",
        "path":   "results/baseline_v3/baseline_train_log_seed0.json",
        "color":  "#e6550d",
    },
    {
        "label":  "Comm-4",
        "path":   "results/comm4_v2/comm4_train_log_seed0.json",
        "color":  "#31a354",
    },
    {
        "label":  "Comm-16",
        "path":   "results/comm16_v2/comm16_train_log_seed0.json",
        "color":  "#3182bd",
    },
]


def rolling(values: list[float], window: int) -> np.ndarray:
    arr = np.array(values, dtype=float)
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot(out_path: Path, window: int) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    fig.suptitle(
        "Training Curves — All Conditions (easy_open, seed 0)",
        fontsize=13, fontweight="bold", y=0.99,
    )

    for spec in LOGS:
        log_path = Path(spec["path"])
        if not log_path.exists():
            print(f"  SKIP (not found): {log_path}")
            continue

        with open(log_path) as f:
            log = json.load(f)

        episodes  = [e["episode"]         for e in log]
        cap_raw   = [float(e["captured"])  for e in log]
        steps_raw = [float(e["steps"])     for e in log]
        epsilon   = [e["epsilon"]          for e in log]

        cap_smooth   = rolling(cap_raw,   window)
        steps_smooth = rolling(steps_raw, window)

        lbl   = spec["label"]
        color = spec["color"]

        # Panel 1: capture rate
        axes[0].plot(episodes, cap_raw,    color=color, linewidth=0.4, alpha=0.25)
        axes[0].plot(episodes, cap_smooth, color=color, linewidth=2.0, label=lbl)

        # Panel 2: steps
        axes[1].plot(episodes, steps_raw,    color=color, linewidth=0.4, alpha=0.25)
        axes[1].plot(episodes, steps_smooth, color=color, linewidth=2.0, label=lbl)

        # Panel 3: epsilon
        axes[2].plot(episodes, epsilon, color=color, linewidth=1.5, label=lbl)

    # Panel 1 formatting
    ax = axes[0]
    ax.set_ylabel("Capture Rate")
    ax.set_ylim(-0.05, 1.10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Capture Rate ({window}-ep rolling avg)", fontsize=10)

    # Panel 2 formatting
    ax = axes[1]
    ax.set_ylabel("Steps to End")
    ax.axhline(300, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="max (300)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Steps per Episode ({window}-ep rolling avg)", fontsize=10)

    # Panel 3 formatting
    ax = axes[2]
    ax.set_ylabel("Epsilon")
    ax.set_xlabel("Episode")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Epsilon Decay", fontsize=10)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot training curves for all conditions")
    p.add_argument("--out",    default="results/comm_training_curves.png")
    p.add_argument("--window", type=int, default=100)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot(Path(args.out), args.window)
