"""Checkpoint sweep: find the best checkpoint per run, then eval on all maps.

For each condition/seed, evaluates all saved 500-episode checkpoints on easy_open
(the training map), selects the best by:
  1. Highest greedy capture rate
  2. Lowest avg_steps as tie-breaker

Then re-evaluates each selected checkpoint on center_block and split_barrier.

Outputs:
  results/sweep_selection.json   : which checkpoint was chosen per run and why
  results/sweep_raw.json         : all eval results for selected checkpoints
  results/sweep_summary.json     : mean ± std per condition × map
  results/sweep_summary.md       : markdown table

Usage:
    python checkpoint_sweep.py
    python checkpoint_sweep.py --results-dir results --episodes 200 --seed 99
    python checkpoint_sweep.py --skip-selection          # re-use sweep_selection.json
    python checkpoint_sweep.py --only-subdir baseline_s3 # sweep one new run, merge, re-eval all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from eval import run_eval


# ---------------------------------------------------------------------------
# Run registry — same structure as eval_all_seeds.py
# ---------------------------------------------------------------------------

RUNS = [
    # (condition, subdir, vocab_size_or_none, ep_prefix)
    ("No-Comm",  "baseline_v3",  None, "baseline"),
    ("No-Comm",  "baseline_s1",  None, "baseline"),
    ("No-Comm",  "baseline_s3",  None, "baseline"),
    ("Comm-4",   "comm4_v2",     4,    "comm4"),
    ("Comm-4",   "comm4_s1",     4,    "comm4"),
    ("Comm-4",   "comm4_s2",     4,    "comm4"),
    ("Comm-16",  "comm16_v2",    16,   "comm16"),
    ("Comm-16",  "comm16_s1",    16,   "comm16"),
    ("Comm-16",  "comm16_s2",    16,   "comm16"),
]

SEED_MAP = {
    "baseline_v3": 0, "baseline_s1": 1, "baseline_s3": 3,
    "comm4_v2":    0, "comm4_s1":    1, "comm4_s2":    2,
    "comm16_v2":   0, "comm16_s1":   1, "comm16_s2":   2,
}

MAPS = ["easy_open", "center_block", "split_barrier", "large_split"]
METRICS = ["capture_rate", "avg_steps", "collision_rate"]


# ---------------------------------------------------------------------------
# Phase 1: sweep all checkpoints on easy_open, pick best per run
# ---------------------------------------------------------------------------

def sweep_checkpoints(
    results_dir: Path,
    eval_episodes: int,
    eval_seed: int,
) -> list[dict]:
    """For each run, evaluate all 500-ep checkpoints on easy_open and pick best."""
    selections = []

    for condition, subdir, vocab, prefix in RUNS:
        ckpt_dir = results_dir / subdir / "checkpoints"
        if not ckpt_dir.exists():
            print(f"  SKIP (not found): {ckpt_dir}")
            continue

        # Find all periodic checkpoints (not final)
        ckpts = sorted(ckpt_dir.glob(f"{prefix}_ep*.pt"))
        if not ckpts:
            print(f"  SKIP (no ep checkpoints): {ckpt_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"Sweeping {condition} | {subdir} ({len(ckpts)} checkpoints)")
        print(f"{'='*60}")

        best_ckpt   = None
        best_cap    = -1.0
        best_steps  = float("inf")
        best_ep     = None
        all_scores  = []

        for ckpt in ckpts:
            # Extract episode number from filename
            name = ckpt.stem  # e.g. "baseline_ep002000"
            try:
                ep = int(name.split("ep")[-1])
            except ValueError:
                continue

            class Args:
                checkpoint    = str(ckpt)
                random_policy = False
                map           = "easy_open"
                episodes      = eval_episodes
                seed          = eval_seed
                n_catch       = None
                output        = None

            try:
                metrics = run_eval(Args())
                cap   = metrics["capture_rate"]
                steps = metrics["avg_steps"]
                all_scores.append({"ep": ep, "capture_rate": cap, "avg_steps": steps})

                # Select: highest capture rate, then lowest steps
                if cap > best_cap or (cap == best_cap and steps < best_steps):
                    best_cap   = cap
                    best_steps = steps
                    best_ckpt  = str(ckpt)
                    best_ep    = ep

            except Exception as e:
                print(f"  ERROR on {ckpt.name}: {e}")

        if best_ckpt is None:
            print(f"  WARNING: no valid checkpoint found for {subdir}")
            continue

        print(f"\n  >>> Selected: ep{best_ep:06d}  cap={best_cap:.3f}  steps={best_steps:.1f}")

        selections.append({
            "condition": condition,
            "subdir":    subdir,
            "seed":      SEED_MAP[subdir],
            "vocab":     vocab,
            "selected_ckpt": best_ckpt,
            "selected_ep":   best_ep,
            "selected_cap":  best_cap,
            "selected_steps": best_steps,
            "all_scores":    all_scores,
        })

    return selections


# ---------------------------------------------------------------------------
# Phase 2: eval selected checkpoints on all maps
# ---------------------------------------------------------------------------

def eval_selected(
    selections: list[dict],
    eval_episodes: int,
    eval_seed: int,
) -> list[dict]:
    raw = []
    total = len(selections) * len(MAPS)
    done  = 0

    for sel in selections:
        for map_name in MAPS:
            done += 1
            print(f"\n[{done}/{total}] {sel['condition']} | {sel['subdir']} | {map_name}"
                  f"  (ep{sel['selected_ep']:06d})")

            class Args:
                checkpoint    = sel["selected_ckpt"]
                random_policy = False
                map           = map_name
                episodes      = eval_episodes
                seed          = eval_seed
                n_catch       = None
                output        = None

            try:
                metrics = run_eval(Args())
                raw.append({
                    "condition":    sel["condition"],
                    "subdir":       sel["subdir"],
                    "train_seed":   sel["seed"],
                    "selected_ep":  sel["selected_ep"],
                    "map":          map_name,
                    **metrics,
                })
            except Exception as e:
                print(f"  ERROR: {e}")

    return raw


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def aggregate(raw: list[dict]) -> dict:
    from collections import defaultdict
    buckets: dict = defaultdict(list)

    for r in raw:
        key = (r["condition"], r["map"])
        buckets[key].append({m: r[m] for m in METRICS})

    summary = {}
    conditions = ["No-Comm", "Comm-4", "Comm-16"]
    for cond in conditions:
        for map_name in MAPS:
            key = (cond, map_name)
            vals = buckets.get(key, [])
            if not vals:
                summary[str(key)] = {"n": 0, "note": "missing"}
                continue
            entry = {"n": len(vals)}
            for m in METRICS:
                vs = [v[m] for v in vals if v[m] is not None]
                entry[f"{m}_mean"] = round(float(np.mean(vs)), 4) if vs else None
                entry[f"{m}_std"]  = round(float(np.std(vs)),  4) if len(vs) > 1 else None
            summary[str(key)] = entry

    return summary


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------

def build_markdown(summary: dict, selections: list[dict]) -> str:
    conditions = ["No-Comm", "Comm-4", "Comm-16"]

    lines = [
        "## Multi-Seed Summary Table (Best-Checkpoint Selection)\n",
        "Trained on `easy_open`. Evaluated seed 99, 200 episodes per condition per seed.\n",
        "Checkpoint selected per run: highest greedy capture rate on easy_open, "
        "then lowest avg_steps as tiebreaker.\n",
        "Values shown as mean ± std across 3 seeds.\n",
        "",
        "| Condition | Map | Capture Rate | Avg Steps | Collision Rate | N seeds |",
        "|-----------|-----|:------------:|:---------:|:--------------:|:-------:|",
    ]

    for cond in conditions:
        for map_name in MAPS:
            key = str((cond, map_name))
            e = summary.get(key, {})
            n = e.get("n", 0)

            def fmt_pct(mean, std):
                if mean is None:
                    return "—"
                s = f"{mean*100:.1f}%"
                if std is not None:
                    s += f" ±{std*100:.1f}"
                return s

            def fmt_val(mean, std):
                if mean is None:
                    return "—"
                s = f"{mean:.1f}"
                if std is not None:
                    s += f" ±{std:.1f}"
                return s

            cap  = fmt_pct(e.get("capture_rate_mean"),  e.get("capture_rate_std"))
            stps = fmt_val(e.get("avg_steps_mean"),     e.get("avg_steps_std"))
            col  = fmt_pct(e.get("collision_rate_mean"), e.get("collision_rate_std"))
            lines.append(f"| {cond} | {map_name} | {cap} | {stps} | {col} | {n} |")

        lines.append("|  |  |  |  |  |  |")

    lines += [
        "",
        "---",
        "",
        "## Checkpoint Selection Details\n",
        "| Condition | Subdir | Selected Ep | Cap (easy_open) | Avg Steps |",
        "|-----------|--------|:-----------:|:---------------:|:---------:|",
    ]
    for sel in selections:
        lines.append(
            f"| {sel['condition']} | {sel['subdir']} | "
            f"ep{sel['selected_ep']:06d} | "
            f"{sel['selected_cap']*100:.1f}% | "
            f"{sel['selected_steps']:.1f} |"
        )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Checkpoint sweep and best-checkpoint eval")
    p.add_argument("--results-dir",   default="results")
    p.add_argument("--episodes",      type=int, default=200)
    p.add_argument("--seed",          type=int, default=99)
    p.add_argument("--skip-selection", action="store_true",
                   help="Skip the sweep phase; load existing sweep_selection.json")
    p.add_argument("--only-subdir",   default=None,
                   help="Sweep only this one subdir, merge into existing selection, re-eval all")
    return p.parse_args()


if __name__ == "__main__":
    args     = parse_args()
    rdir     = Path(args.results_dir)
    sel_path = rdir / "sweep_selection.json"

    # --- Phase 1: checkpoint sweep ---
    if args.only_subdir:
        # Sweep just the one new subdir and merge into existing selections
        if not sel_path.exists():
            print(f"ERROR: --only-subdir requires existing {sel_path}. Run full sweep first.")
            sys.exit(1)
        with open(sel_path) as f:
            selections = json.load(f)
        # Remove any existing entry for this subdir
        selections = [s for s in selections if s["subdir"] != args.only_subdir]
        print(f"Phase 1: Sweeping only '{args.only_subdir}' on easy_open...")
        new_sel = sweep_checkpoints(rdir, args.episodes, args.seed)
        new_sel = [s for s in new_sel if s["subdir"] == args.only_subdir]
        selections.extend(new_sel)
        with open(sel_path, "w") as f:
            json.dump(selections, f, indent=2)
        print(f"\nUpdated selections saved → {sel_path}")
    elif args.skip_selection and sel_path.exists():
        print(f"Loading existing selection from {sel_path}")
        with open(sel_path) as f:
            selections = json.load(f)
    else:
        print("Phase 1: Sweeping checkpoints on easy_open to find best per run...")
        selections = sweep_checkpoints(rdir, args.episodes, args.seed)
        with open(sel_path, "w") as f:
            json.dump(selections, f, indent=2)
        print(f"\nSelections saved → {sel_path}")

    # --- Phase 2: eval on all maps ---
    print("\nPhase 2: Evaluating selected checkpoints on all maps...")
    raw = eval_selected(selections, args.episodes, args.seed)

    raw_path = rdir / "sweep_raw.json"
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"\nRaw results saved → {raw_path}")

    # --- Aggregate ---
    summary = aggregate(raw)
    summary_path = rdir / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved → {summary_path}")

    # --- Markdown ---
    md = build_markdown(summary, selections)
    md_path = rdir / "sweep_summary.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Markdown saved → {md_path}")
    print("\n" + md)
