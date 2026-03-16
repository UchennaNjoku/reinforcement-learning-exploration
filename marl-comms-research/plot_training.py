"""Training curve plots for the no-communication baseline.

Reads baseline_train_log_seed0.json and produces a 3-panel figure:
  1. Capture rate (rolling average)
  2. Average steps per episode (rolling average)
  3. Epsilon decay over episodes

Usage:
    python plot_training.py
    python plot_training.py --log results/baseline_v3/baseline_train_log_seed0.json
    python plot_training.py --log results/baseline_v3/baseline_train_log_seed0.json \
                            --out results/baseline_v3/training_curve.png
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


def rolling(values: list[float], window: int) -> np.ndarray:
    arr = np.array(values, dtype=float)
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot(log_path: Path, out_path: Path, window: int) -> None:
    with open(log_path) as f:
        log = json.load(f)

    episodes   = [e["episode"]     for e in log]
    cap_raw    = [float(e["captured"]) for e in log]
    steps_raw  = [e["steps"]       for e in log]
    epsilon    = [e["epsilon"]      for e in log]

    cap_smooth   = rolling(cap_raw,   window)
    steps_smooth = rolling(steps_raw, window)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(
        "No-Communication Baseline — easy_open (seed 0)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # ── Panel 1: capture rate ────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(episodes, cap_raw,    color="#c0d8f0", linewidth=0.5, alpha=0.6, label="per-episode")
    ax.plot(episodes, cap_smooth, color="#2171b5", linewidth=1.8,            label=f"{window}-ep rolling avg")
    ax.set_ylabel("Capture Rate")
    ax.set_ylim(-0.05, 1.10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    # ── Panel 2: steps per episode ───────────────────────────────────────────
    ax = axes[1]
    ax.plot(episodes, steps_raw,    color="#c7e9c0", linewidth=0.5, alpha=0.6, label="per-episode")
    ax.plot(episodes, steps_smooth, color="#238b45", linewidth=1.8,            label=f"{window}-ep rolling avg")
    ax.set_ylabel("Steps to End")
    ax.axhline(300, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="max steps (300)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── Panel 3: epsilon ─────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(episodes, epsilon, color="#d94801", linewidth=1.5)
    ax.set_ylabel("Epsilon")
    ax.set_xlabel("Episode")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot baseline training curves")
    p.add_argument(
        "--log",
        default="results/baseline_v3/baseline_train_log_seed0.json",
        help="Path to training log JSON",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output PNG path (default: same dir as log)",
    )
    p.add_argument(
        "--window",
        type=int,
        default=100,
        help="Rolling average window size",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    log_path = Path(args.log)
    out_path = Path(args.out) if args.out else log_path.parent / "training_curve.png"
    plot(log_path, out_path, args.window)
