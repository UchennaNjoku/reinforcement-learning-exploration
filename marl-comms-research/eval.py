"""Evaluation script for trained pursuit agents.

Supports three policy modes:
  1. Baseline (no-comm) checkpoint  — PursuitQNet
  2. Comm checkpoint                — CommQNet (auto-detected from saved args)
  3. Random policy                  — no checkpoint needed

Runs N episodes with a greedy policy (epsilon = 0) and records:
  - capture_rate      : fraction of episodes where prey was caught
  - escape_rate       : fraction where prey was NOT caught
  - avg_steps         : mean episode length across all episodes
  - avg_steps_capture : mean steps for episodes that ended in capture
  - collision_rate    : fraction of agent steps that were blocked moves

Output is printed to stdout and optionally saved as JSON.

Usage:
    python eval.py --checkpoint results/baseline_v3/checkpoints/baseline_final.pt
    python eval.py --checkpoint results/comm/checkpoints/comm4_final.pt
    python eval.py --random-policy --map easy_open
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from envs import make_fixed_pursuit_env
from models.qnet import PursuitQNet
from models.comm_qnet import CommQNet
from train_comm import (
    zero_messages, build_received, msg_to_onehot, AGENT_IDS, N_AGENTS
)


# ---------------------------------------------------------------------------
# Action selectors
# ---------------------------------------------------------------------------

def greedy_baseline(
    qnet: PursuitQNet,
    observations: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, int]:
    actions: dict[str, int] = {}
    for agent, obs in observations.items():
        obs_t = torch.from_numpy(obs.transpose(2, 0, 1)).unsqueeze(0).to(device)
        aid_t = torch.tensor([AGENT_IDS[agent]], dtype=torch.long, device=device)
        with torch.no_grad():
            q = qnet(obs_t, aid_t)
        actions[agent] = int(q.argmax(dim=1).item())
    return actions


def greedy_comm(
    qnet: CommQNet,
    observations: dict[str, np.ndarray],
    prev_messages: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    """Returns move actions and updated message dict using separate heads."""
    move_actions: dict[str, int] = {}
    new_messages: dict[str, np.ndarray] = {}

    for agent, obs in observations.items():
        obs_t  = torch.from_numpy(obs.transpose(2, 0, 1)).unsqueeze(0).to(device)
        aid_t  = torch.tensor([AGENT_IDS[agent]], dtype=torch.long, device=device)
        recv_t = torch.from_numpy(
            build_received(agent, prev_messages)
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            move_q, msg_q = qnet(obs_t, aid_t, recv_t)
        move = int(move_q.argmax(dim=1).item())
        msg  = int(msg_q.argmax(dim=1).item())
        move_actions[agent] = move
        new_messages[agent] = msg_to_onehot(msg, qnet.vocab_size)

    return move_actions, new_messages


def random_actions(env, observations: dict[str, np.ndarray]) -> dict[str, int]:
    return {agent: env.action_space(agent).sample() for agent in observations}


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def load_checkpoint(path: str, device: torch.device):
    """Load checkpoint, auto-detect model type from saved args.

    Returns (model, ckpt_dict, is_comm).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    vocab_size = saved_args.get("vocab_size", None)

    if vocab_size is not None:
        # Communication model
        model = CommQNet(
            n_agents=N_AGENTS,
            n_move_actions=5,
            vocab_size=vocab_size,
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, ckpt, True
    else:
        # Baseline model
        model = PursuitQNet(n_agents=N_AGENTS, n_actions=5).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, ckpt, False


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_eval(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = None
    is_comm = False

    ckpt = None
    if not args.random_policy:
        model, ckpt, is_comm = load_checkpoint(args.checkpoint, device)
        trained_ep    = ckpt.get("episode", "?")
        trained_steps = ckpt.get("total_steps", "?")
        model_type    = "comm" if is_comm else "baseline"
        print(
            f"Loaded {model_type} checkpoint: "
            f"episode={trained_ep}, total_steps={trained_steps}"
        )
        if is_comm:
            print(f"  vocab_size={model.vocab_size}")
    else:
        print("Evaluating RANDOM policy (no checkpoint).")

    # Resolve map and n_catch: prefer checkpoint's saved args, fall back to CLI.
    # Warn loudly if the CLI explicitly differs from the checkpoint.
    saved = ckpt.get("args", {}) if ckpt else {}
    ckpt_map     = saved.get("map",     None)
    ckpt_n_catch = saved.get("n_catch", None)

    # args.map / args.n_catch are None when the user didn't pass them explicitly.
    # Priority: explicit CLI flag > checkpoint saved args > hardcoded default.
    eval_map     = args.map     if args.map     is not None else (ckpt_map     or "easy_open")
    eval_n_catch = args.n_catch if args.n_catch is not None else (ckpt_n_catch or 1)

    if args.map is None and ckpt_map is not None:
        print(f"  Using checkpoint's map='{ckpt_map}'.")
    if args.map is not None and ckpt_map is not None and args.map != ckpt_map:
        print(
            f"  NOTE: evaluating on map='{args.map}' "
            f"(checkpoint trained on '{ckpt_map}') — cross-map eval."
        )

    env = make_fixed_pursuit_env(map_name=eval_map, n_catch=eval_n_catch, surround=False)

    captures = 0
    episode_steps: list[int] = []
    capture_steps: list[int] = []
    total_blocked      = 0
    total_agent_steps  = 0

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_steps = 0
        captured = False

        if is_comm:
            prev_messages = zero_messages(model.vocab_size)

        while True:
            if model is None:
                actions = random_actions(env, obs)
            elif is_comm:
                actions, prev_messages = greedy_comm(model, obs, prev_messages, device)
            else:
                actions = greedy_baseline(model, obs, device)

            next_obs, _, terminations, truncations, infos = env.step(actions)
            ep_steps += 1

            for agent, info in infos.items():
                if info.get("blocked_move", False):
                    total_blocked += 1
                total_agent_steps += 1
                if info.get("evaders_remaining", 1) == 0:
                    captured = True

            done = any(terminations.values()) or any(truncations.values()) or not env.agents
            obs  = next_obs

            if done:
                break

        episode_steps.append(ep_steps)
        if captured:
            captures += 1
            capture_steps.append(ep_steps)

    env.close()

    n = args.episodes
    capture_rate      = captures / n
    avg_steps         = float(np.mean(episode_steps))
    avg_steps_capture = float(np.mean(capture_steps)) if capture_steps else None
    collision_rate    = total_blocked / max(1, total_agent_steps)

    if args.random_policy:
        policy_label = "random"
    elif is_comm:
        saved_args = ckpt.get("args", {})
        policy_label = f"comm{model.vocab_size}"
    else:
        policy_label = "baseline"

    metrics = {
        "map":               eval_map,
        "episodes":          n,
        "seed":              args.seed,
        "policy":            policy_label,
        "capture_rate":      round(capture_rate, 4),
        "escape_rate":       round(1.0 - capture_rate, 4),
        "avg_steps":         round(avg_steps, 2),
        "avg_steps_capture": round(avg_steps_capture, 2) if avg_steps_capture else None,
        "collision_rate":    round(collision_rate, 4),
    }

    print("\n--- Evaluation Results ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved → {out_path}")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained pursuit agent")
    p.add_argument("--checkpoint",    default=None,
                   help="Path to .pt checkpoint (baseline or comm; auto-detected)")
    p.add_argument("--random-policy", action="store_true",
                   help="Use random actions (no checkpoint needed)")
    p.add_argument("--map",           default=None,
                   choices=["easy_open", "center_block", "split_barrier"],
                   help="Map to evaluate on. Defaults to checkpoint's training map.")
    p.add_argument("--episodes",      type=int, default=50)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--n-catch",       type=int, default=None,
                   help="Pursuers required on evader cell. Defaults to checkpoint's value.")
    p.add_argument("--output",        default=None,
                   help="Path to save JSON results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.random_policy and not args.checkpoint:
        print("Error: provide --checkpoint or --random-policy")
        sys.exit(1)
    run_eval(args)
