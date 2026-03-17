"""Evaluate all seeds for all conditions and produce a summary table.

Runs greedy evaluation for every checkpoint found under results/,
aggregates mean ± std across seeds per condition, and saves:
  - results/all_seeds_raw.json     : every individual eval result
  - results/summary_table.json     : mean ± std per condition × map
  - results/summary_table.md       : markdown table for the paper

Usage:
    python eval_all_seeds.py                        # uses default results/ dir
    python eval_all_seeds.py --results-dir results/ # explicit path
    python eval_all_seeds.py --episodes 200 --seed 99
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
# Checkpoint registry
# Defines the canonical set of checkpoints to evaluate.
# Add entries here as new seeds are downloaded.
# ---------------------------------------------------------------------------

def build_registry(results_dir: Path) -> list[dict]:
    """Return list of {label, condition, seed, checkpoint_path} dicts."""
    registry = []

    specs = [
        # (condition_label, glob_pattern_for_results_subdir, vocab_size_or_none)
        ("No-Comm",  "baseline_v3",  None),
        ("No-Comm",  "baseline_s1",  None),
        ("No-Comm",  "baseline_s2",  None),
        ("Comm-4",   "comm4_v2",     4),
        ("Comm-4",   "comm4_s1",     4),
        ("Comm-4",   "comm4_s2",     4),
        ("Comm-16",  "comm16_v2",    16),
        ("Comm-16",  "comm16_s1",    16),
        ("Comm-16",  "comm16_s2",    16),
    ]

    seed_map = {
        "baseline_v3": 0, "baseline_s1": 1, "baseline_s2": 2,
        "comm4_v2":    0, "comm4_s1":    1, "comm4_s2":    2,
        "comm16_v2":   0, "comm16_s1":   1, "comm16_s2":   2,
    }

    for condition, subdir, vocab in specs:
        ckpt_dir = results_dir / subdir / "checkpoints"
        if not ckpt_dir.exists():
            print(f"  SKIP (not found): {ckpt_dir}")
            continue

        prefix = "baseline_final" if vocab is None else f"comm{vocab}_final"
        ckpt = ckpt_dir / f"{prefix}.pt"
        if not ckpt.exists():
            print(f"  SKIP (no final ckpt): {ckpt}")
            continue

        registry.append({
            "condition": condition,
            "subdir":    subdir,
            "seed":      seed_map[subdir],
            "ckpt":      str(ckpt),
        })

    return registry


MAPS = ["easy_open", "center_block", "split_barrier"]


# ---------------------------------------------------------------------------
# Run evals
# ---------------------------------------------------------------------------

def collect_results(
    registry: list[dict],
    eval_episodes: int,
    eval_seed: int,
) -> list[dict]:
    """Run eval for every (checkpoint, map) pair and return raw results."""
    raw = []
    total = len(registry) * len(MAPS)
    done  = 0

    for entry in registry:
        for map_name in MAPS:
            done += 1
            print(f"\n[{done}/{total}] {entry['condition']} | {entry['subdir']} | {map_name}")

            # Build a minimal args namespace for run_eval
            class Args:
                checkpoint    = entry["ckpt"]
                random_policy = False
                map           = map_name
                episodes      = eval_episodes
                seed          = eval_seed
                n_catch       = None
                output        = None

            try:
                metrics = run_eval(Args())
                raw.append({
                    "condition": entry["condition"],
                    "subdir":    entry["subdir"],
                    "train_seed":entry["seed"],
                    "map":       map_name,
                    **metrics,
                })
            except Exception as e:
                print(f"  ERROR: {e}")

    return raw


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

METRICS = ["capture_rate", "avg_steps", "collision_rate"]

def aggregate(raw: list[dict]) -> dict:
    """Compute mean ± std per (condition, map) across seeds."""
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

def build_markdown(summary: dict) -> str:
    conditions = ["No-Comm", "Comm-4", "Comm-16"]
    map_labels = {"easy_open": "easy_open", "center_block": "center_block", "split_barrier": "split_barrier"}

    lines = [
        "## Multi-Seed Summary Table\n",
        "Trained on `easy_open`. Evaluated seed 99, 200 episodes per condition per seed.\n",
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

            cap  = fmt_pct(e.get("capture_rate_mean"),   e.get("capture_rate_std"))
            stps = fmt_val(e.get("avg_steps_mean"),       e.get("avg_steps_std"))
            col  = fmt_pct(e.get("collision_rate_mean"),  e.get("collision_rate_std"))

            lines.append(f"| {cond} | {map_labels[map_name]} | {cap} | {stps} | {col} | {n} |")

        lines.append("|  |  |  |  |  |  |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate all seeds and summarize")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--episodes",    type=int, default=200)
    p.add_argument("--seed",        type=int, default=99)
    p.add_argument("--skip-eval",   action="store_true",
                   help="Skip running eval, just re-aggregate from all_seeds_raw.json")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    rdir   = Path(args.results_dir)
    raw_path = rdir / "all_seeds_raw.json"

    if args.skip_eval and raw_path.exists():
        print(f"Loading existing raw results from {raw_path}")
        with open(raw_path) as f:
            raw = json.load(f)
    else:
        registry = build_registry(rdir)
        print(f"\nFound {len(registry)} checkpoints to evaluate.")
        if not registry:
            print("No checkpoints found. Check --results-dir.")
            sys.exit(1)
        raw = collect_results(registry, args.episodes, args.seed)
        with open(raw_path, "w") as f:
            json.dump(raw, f, indent=2)
        print(f"\nRaw results saved → {raw_path}")

    summary = aggregate(raw)
    summary_path = rdir / "summary_table.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved → {summary_path}")

    md = build_markdown(summary)
    md_path = rdir / "summary_table.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Markdown table saved → {md_path}")
    print("\n" + md)
