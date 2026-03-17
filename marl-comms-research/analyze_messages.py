"""Interpretability analysis of learned communication messages.

Reads comm4_msg_log_seed0.json and produces:
  1. results/msg_analysis/freq_dist.png      — symbol frequency bar chart
  2. results/msg_analysis/agent_freq.png     — per-agent symbol usage heatmap
  3. results/msg_analysis/temporal.png       — symbol usage over episode timestep
  4. results/msg_analysis/message_analysis.md — text summary of findings

Usage:
    python analyze_messages.py
    python analyze_messages.py --log results/comm4_v2/comm4_msg_log_seed0.json
    python analyze_messages.py --vocab-size 16 --log results/comm16_v2/comm16_msg_log_seed0.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


AGENTS = ["pursuer_0", "pursuer_1", "pursuer_2"]
AGENT_SHORT = ["P0", "P1", "P2"]


# ---------------------------------------------------------------------------
# Load + flatten
# ---------------------------------------------------------------------------

def load_log(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def flatten_messages(log: list[dict]) -> tuple[list[int], dict[str, list[int]]]:
    """Return all messages (flat) and per-agent message lists."""
    all_msgs: list[int] = []
    per_agent: dict[str, list[int]] = {a: [] for a in AGENTS}

    for ep in log:
        for step_msgs in ep["messages"]:
            for agent in AGENTS:
                sym = step_msgs.get(agent, -1)
                if sym >= 0:
                    all_msgs.append(sym)
                    per_agent[agent].append(sym)

    return all_msgs, per_agent


# ---------------------------------------------------------------------------
# Figure 1: Overall symbol frequency
# ---------------------------------------------------------------------------

def plot_freq_dist(all_msgs: list[int], vocab_size: int, out_dir: Path) -> dict:
    counts = Counter(all_msgs)
    symbols = list(range(vocab_size))
    freqs   = [counts.get(s, 0) for s in symbols]
    total   = sum(freqs)
    pcts    = [100 * f / total for f in freqs]

    expected = 100 / vocab_size
    entropy  = -sum((p/100) * np.log2(p/100 + 1e-12) for p in pcts)
    max_entropy = np.log2(vocab_size)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([str(s) for s in symbols], pcts, color="#4292c6", edgecolor="white")
    ax.axhline(expected, color="#d94801", linestyle="--", linewidth=1.2,
               label=f"Uniform ({expected:.1f}%)")
    for bar, p in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{p:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Symbol")
    ax.set_ylabel("Usage (%)")
    ax.set_title(
        f"Comm-{vocab_size}: Symbol Frequency Distribution\n"
        f"(entropy = {entropy:.2f} bits, max = {max_entropy:.2f} bits, "
        f"utilization = {entropy/max_entropy*100:.0f}%)",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = out_dir / "freq_dist.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")

    return {
        "total_messages": total,
        "per_symbol_pct": {str(s): round(pcts[i], 2) for i, s in enumerate(symbols)},
        "entropy_bits": round(float(entropy), 4),
        "max_entropy_bits": round(float(max_entropy), 4),
        "utilization_pct": round(float(entropy / max_entropy * 100), 1),
        "dominant_symbol": str(int(np.argmax(pcts))),
        "dominant_pct": round(float(max(pcts)), 1),
    }


# ---------------------------------------------------------------------------
# Figure 2: Per-agent symbol usage heatmap
# ---------------------------------------------------------------------------

def plot_agent_freq(per_agent: dict[str, list[int]], vocab_size: int, out_dir: Path) -> dict:
    matrix = np.zeros((len(AGENTS), vocab_size))
    for i, agent in enumerate(AGENTS):
        counts = Counter(per_agent[agent])
        total  = max(1, sum(counts.values()))
        for s in range(vocab_size):
            matrix[i, s] = counts.get(s, 0) / total * 100

    fig, ax = plt.subplots(figsize=(max(6, vocab_size * 0.8), 3.5))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=matrix.max())
    ax.set_xticks(range(vocab_size))
    ax.set_xticklabels([str(s) for s in range(vocab_size)])
    ax.set_yticks(range(len(AGENTS)))
    ax.set_yticklabels(AGENT_SHORT)
    ax.set_xlabel("Symbol")
    ax.set_title(f"Comm-{vocab_size}: Per-Agent Symbol Usage (%)", fontsize=10)

    for i in range(len(AGENTS)):
        for j in range(vocab_size):
            val = matrix[i, j]
            color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="%")
    fig.tight_layout()
    out = out_dir / "agent_freq.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")

    # Measure role differentiation: std dev of per-agent distributions
    row_entropy = []
    for i in range(len(AGENTS)):
        p = matrix[i] / 100
        h = -np.sum(p * np.log2(p + 1e-12))
        row_entropy.append(round(float(h), 4))

    inter_agent_variation = round(float(np.std(matrix, axis=0).mean()), 2)

    return {
        "per_agent_entropy_bits": dict(zip(AGENT_SHORT, row_entropy)),
        "inter_agent_variation_avg_std": inter_agent_variation,
    }


# ---------------------------------------------------------------------------
# Figure 3: Symbol usage vs episode timestep (early / mid / late step)
# ---------------------------------------------------------------------------

def plot_temporal(log: list[dict], vocab_size: int, out_dir: Path) -> dict:
    """How does symbol usage shift across timestep positions within an episode?"""
    # Bin steps into thirds: early (0-33%), mid (33-66%), late (66-100%)
    bins = {"early": defaultdict(int), "mid": defaultdict(int), "late": defaultdict(int)}

    for ep in log:
        msgs = ep["messages"]
        n = len(msgs)
        if n == 0:
            continue
        for t, step_msgs in enumerate(msgs):
            frac = t / n
            if frac < 0.33:
                bucket = "early"
            elif frac < 0.66:
                bucket = "mid"
            else:
                bucket = "late"
            for agent in AGENTS:
                sym = step_msgs.get(agent, -1)
                if sym >= 0:
                    bins[bucket][sym] += 1

    symbols = list(range(vocab_size))
    bucket_labels = ["Early\n(0–33%)", "Mid\n(33–66%)", "Late\n(66–100%)"]
    colors = ["#9ecae1", "#4292c6", "#084594"]

    fig, ax = plt.subplots(figsize=(max(7, vocab_size), 4))
    x = np.arange(vocab_size)
    width = 0.25
    offsets = [-width, 0, width]

    for i, (bucket, label, color) in enumerate(zip(["early", "mid", "late"], bucket_labels, colors)):
        total = max(1, sum(bins[bucket].values()))
        pcts  = [bins[bucket].get(s, 0) / total * 100 for s in symbols]
        ax.bar(x + offsets[i], pcts, width, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in symbols])
    ax.set_xlabel("Symbol")
    ax.set_ylabel("Usage (%)")
    ax.set_title(
        f"Comm-{vocab_size}: Symbol Usage by Episode Phase\n"
        "(Does symbol meaning shift with time-into-episode?)",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = out_dir / "temporal.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")

    # Measure temporal consistency: max change in dominant symbol share
    dominant_shifts = {}
    for s in symbols:
        totals = {b: max(1, sum(bins[b].values())) for b in bins}
        pcts_by_phase = [bins[b].get(s, 0) / totals[b] * 100
                         for b in ["early", "mid", "late"]]
        dominant_shifts[str(s)] = round(float(max(pcts_by_phase) - min(pcts_by_phase)), 1)

    return {"temporal_shift_pct": dominant_shifts}


# ---------------------------------------------------------------------------
# Figure 4: Capture vs non-capture symbol distributions
# ---------------------------------------------------------------------------

MIN_ESCAPED_EPISODES = 20  # minimum escaped episodes required for capture/escape analysis


def analyze_capture_correlation(log: list[dict], vocab_size: int, min_escaped: int = MIN_ESCAPED_EPISODES) -> dict:
    """Do messages differ between episodes that end in capture vs escape?

    Returns a dict with 'insufficient_data': True if too few escaped episodes
    exist to draw reliable conclusions.
    """
    n_captured = sum(1 for ep in log if ep["captured"])
    n_escaped  = sum(1 for ep in log if not ep["captured"])

    if n_escaped < min_escaped:
        return {
            "insufficient_data": True,
            "n_captured": n_captured,
            "n_escaped":  n_escaped,
            "min_required": min_escaped,
        }

    counts = {"captured": Counter(), "escaped": Counter()}
    for ep in log:
        key = "captured" if ep["captured"] else "escaped"
        for step_msgs in ep["messages"]:
            for agent in AGENTS:
                sym = step_msgs.get(agent, -1)
                if sym >= 0:
                    counts[key][sym] += 1

    result = {"insufficient_data": False, "n_captured": n_captured, "n_escaped": n_escaped}
    for outcome in ["captured", "escaped"]:
        total = max(1, sum(counts[outcome].values()))
        result[outcome] = {
            str(s): round(counts[outcome].get(s, 0) / total * 100, 1)
            for s in range(vocab_size)
        }

    divergence = {}
    for s in range(vocab_size):
        cap_pct = result["captured"].get(str(s), 0)
        esc_pct = result["escaped"].get(str(s), 0)
        divergence[str(s)] = round(abs(cap_pct - esc_pct), 1)

    result["max_divergence_symbol"] = str(max(divergence, key=divergence.__getitem__))
    result["max_divergence_pct"]    = max(divergence.values())

    return result


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def build_report(
    vocab_size: int,
    log_path: Path,
    freq_stats: dict,
    agent_stats: dict,
    temporal_stats: dict,
    capture_stats: dict,
) -> str:
    n_ep_analyzed = freq_stats["total_messages"]

    lines = [
        f"# Comm-{vocab_size} Message Interpretability Analysis\n",
        f"**Source:** `{log_path}`  ",
        f"**Total messages analyzed:** {n_ep_analyzed:,}  ",
        f"**Vocabulary size:** {vocab_size} symbols\n",
        "---\n",
        "## 1. Symbol Frequency Distribution\n",
        f"Entropy: **{freq_stats['entropy_bits']} bits** "
        f"(max possible = {freq_stats['max_entropy_bits']} bits, "
        f"utilization = **{freq_stats['utilization_pct']}%**)  \n",
        "Symbol usage percentages:\n",
    ]
    for sym, pct in freq_stats["per_symbol_pct"].items():
        lines.append(f"- Symbol {sym}: {pct}%")
    lines.append("")

    uniform_pct = round(100 / vocab_size, 1)
    dominant_sym = freq_stats["dominant_symbol"]
    dominant_pct = freq_stats["dominant_pct"]
    utilization  = freq_stats["utilization_pct"]

    if utilization >= 90:
        interp = (
            f"The vocabulary is used nearly uniformly ({utilization}% of max entropy). "
            f"No symbol dominates — agents are distributing meaning across all {vocab_size} symbols."
        )
    elif utilization >= 65:
        interp = (
            f"Symbol {dominant_sym} is used most often ({dominant_pct}% vs {uniform_pct}% uniform), "
            f"but the channel is still reasonably utilized ({utilization}% of max entropy). "
            "Some symbols are preferred but all are in use."
        )
    else:
        interp = (
            f"The channel is underutilized ({utilization}% of max entropy). "
            f"Symbol {dominant_sym} dominates at {dominant_pct}%, suggesting agents converged "
            "on a small effective vocabulary."
        )

    lines += [f"\n**Interpretation:** {interp}\n", "---\n"]

    lines += [
        "## 2. Per-Agent Symbol Usage\n",
        f"Average inter-agent variation (std across agents per symbol): "
        f"**{agent_stats['inter_agent_variation_avg_std']}%**\n",
        "Per-agent channel entropy (bits):\n",
    ]
    for agent, h in agent_stats["per_agent_entropy_bits"].items():
        lines.append(f"- {agent}: {h} bits")
    lines.append("")

    variation = agent_stats["inter_agent_variation_avg_std"]
    if variation < 2.0:
        role_interp = (
            "All three agents use symbols in very similar proportions — "
            "no role differentiation is evident. The shared policy has not specialized "
            "different agents to send different message types."
        )
    elif variation < 5.0:
        role_interp = (
            "Mild inter-agent variation exists. Agents share similar preferences "
            "but show some differentiation, possibly due to different typical positions "
            "relative to the prey."
        )
    else:
        role_interp = (
            f"Notable inter-agent variation ({variation}%). Different agents have developed "
            "distinct symbol preferences despite sharing the same network parameters — "
            "position-dependent specialization is likely."
        )

    lines += [f"\n**Interpretation:** {role_interp}\n", "---\n"]

    lines += [
        "## 3. Temporal Analysis (Symbol Usage by Episode Phase)\n",
        "Maximum phase shift per symbol (early vs late usage %):\n",
    ]
    for sym, shift in temporal_stats["temporal_shift_pct"].items():
        lines.append(f"- Symbol {sym}: Δ{shift}%")
    max_shift = max(temporal_stats["temporal_shift_pct"].values())
    lines.append("")

    if max_shift < 5:
        temp_interp = (
            "Symbol usage is nearly uniform across episode phases. "
            "Messages do not encode time-into-episode information."
        )
    elif max_shift < 15:
        temp_interp = (
            f"Moderate temporal variation (max Δ{max_shift}%). "
            "Some symbols shift in frequency as episodes progress, "
            "possibly reflecting changing coordination needs (spread-out early, converge late)."
        )
    else:
        temp_interp = (
            f"Strong temporal variation (max Δ{max_shift}%). "
            "Symbol meaning appears to change significantly across episode phases — "
            "this could encode pursuit stage information (searching vs closing in)."
        )

    lines += [f"\n**Interpretation:** {temp_interp}\n", "---\n"]

    lines += ["## 4. Capture vs Escape Correlation\n"]

    if capture_stats.get("insufficient_data"):
        n_esc = capture_stats["n_escaped"]
        n_cap = capture_stats["n_captured"]
        min_req = capture_stats["min_required"]
        lines += [
            f"**Not computed** — only {n_esc} escaped episode(s) out of "
            f"{n_cap + n_esc} total (minimum required: {min_req}).\n",
            "The model almost never fails on this map/checkpoint combination, "
            "so there is insufficient data to compare message distributions between "
            "captured and escaped episodes. Re-run on a harder map or a weaker checkpoint "
            "to obtain a meaningful sample of failures.\n",
            "---\n",
        ]
        cap_interp_summary = f"not computed ({n_esc} escaped ep < {min_req} required)"
        max_div_sym = "—"
        max_div_pct = None
    else:
        n_esc = capture_stats["n_escaped"]
        n_cap = capture_stats["n_captured"]
        lines += [
            f"Based on {n_cap} captured and {n_esc} escaped episodes.\n",
            "Symbol usage in captured vs escaped episodes:\n",
            "| Symbol | Captured (%) | Escaped (%) | Δ |",
            "|--------|:------------:|:-----------:|:--:|",
        ]
        for s in range(vocab_size):
            cap = capture_stats["captured"].get(str(s), 0)
            esc = capture_stats["escaped"].get(str(s), 0)
            delta = abs(cap - esc)
            lines.append(f"| {s} | {cap} | {esc} | {delta:.1f} |")

        max_div_sym = capture_stats["max_divergence_symbol"]
        max_div_pct = capture_stats["max_divergence_pct"]
        lines.append("")

        if max_div_pct < 3:
            cap_interp = (
                "Symbol usage is essentially identical in captured vs escaped episodes. "
                "Messages do not differentially signal success vs failure outcomes."
            )
        elif max_div_pct < 8:
            cap_interp = (
                f"Symbol {max_div_sym} shows the largest divergence ({max_div_pct}% difference) "
                "between captured and escaped episodes. Weak correlation exists but is not strongly "
                "predictive of outcome."
            )
        else:
            cap_interp = (
                f"Symbol {max_div_sym} diverges by {max_div_pct}% between captured and escaped episodes. "
                "This symbol is meaningfully associated with coordination outcomes — "
                "it may encode 'I see the prey' or a convergence signal."
            )

        lines += [f"\n**Interpretation:** {cap_interp}\n", "---\n"]
        cap_interp_summary = f"{max_div_pct}% (symbol {max_div_sym})"

    lines += [
        "## 5. Summary\n",
        f"| Metric | Value |",
        "|--------|-------|",
        f"| Vocab size | {vocab_size} |",
        f"| Channel entropy | {freq_stats['entropy_bits']} / {freq_stats['max_entropy_bits']} bits ({freq_stats['utilization_pct']}%) |",
        f"| Dominant symbol | {dominant_sym} ({dominant_pct}%) |",
        f"| Inter-agent variation | {agent_stats['inter_agent_variation_avg_std']}% avg std |",
        f"| Max temporal shift | {max(temporal_stats['temporal_shift_pct'].values())}% |",
        f"| Max capture/escape divergence | {cap_interp_summary} |",
    ]

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze learned communication messages")
    p.add_argument("--log",          default="results/comm4_v2/comm4_msg_log_seed0.json")
    p.add_argument("--vocab-size",   type=int, default=4)
    p.add_argument("--out-dir",      default="results/msg_analysis")
    p.add_argument("--min-escaped",  type=int, default=MIN_ESCAPED_EPISODES,
                   help="Minimum escaped episodes required for capture/escape analysis")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    log_path = Path(args.log)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {log_path} ...")
    log = load_log(log_path)
    print(f"  {len(log)} episodes loaded.")

    all_msgs, per_agent = flatten_messages(log)
    print(f"  {len(all_msgs):,} total messages.")

    print("\n[1/4] Symbol frequency distribution ...")
    freq_stats = plot_freq_dist(all_msgs, args.vocab_size, out_dir)

    print("\n[2/4] Per-agent symbol usage ...")
    agent_stats = plot_agent_freq(per_agent, args.vocab_size, out_dir)

    print("\n[3/4] Temporal analysis ...")
    temporal_stats = plot_temporal(log, args.vocab_size, out_dir)

    print("\n[4/4] Capture correlation ...")
    capture_stats = analyze_capture_correlation(log, args.vocab_size, args.min_escaped)

    report = build_report(
        args.vocab_size, log_path,
        freq_stats, agent_stats, temporal_stats, capture_stats,
    )
    report_path = out_dir / "message_analysis.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved → {report_path}")
    print("\n" + report)
