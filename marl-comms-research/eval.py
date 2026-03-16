"""Evaluation script for trained pursuit agents.

Runs N episodes with a greedy policy (epsilon = 0) and records:
  - capture_rate      : fraction of episodes where prey was caught
  - escape_rate       : fraction where prey was NOT caught
  - avg_steps         : mean episode length across all episodes
  - avg_steps_capture : mean steps for episodes that ended in capture
  - collision_rate    : fraction of agent steps that were blocked moves

Output is printed to stdout and saved as JSON.

Usage:
    python eval.py --checkpoint results/baseline/checkpoints/baseline_final.pt
    python eval.py --checkpoint path/to/ckpt.pt --episodes 100 --map easy_open
    python eval.py --random-policy --map easy_open   # random baseline (no checkpoint)
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


AGENT_IDS: dict[str, int] = {
    "pursuer_0": 0,
    "pursuer_1": 1,
    "pursuer_2": 2,
}


def greedy_actions(
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


def random_actions(
    env,
    observations: dict[str, np.ndarray],
) -> dict[str, int]:
    return {agent: env.action_space(agent).sample() for agent in observations}


def run_eval(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    qnet = None
    if not args.random_policy:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        qnet = PursuitQNet(n_agents=3, n_actions=5).to(device)
        qnet.load_state_dict(ckpt["model_state"])
        qnet.eval()
        trained_ep = ckpt.get("episode", "?")
        trained_steps = ckpt.get("total_steps", "?")
        print(f"Loaded checkpoint: episode={trained_ep}, total_steps={trained_steps}")
    else:
        print("Evaluating RANDOM policy (no checkpoint).")

    env = make_fixed_pursuit_env(map_name=args.map)
    n_actions = env.action_space("pursuer_0").n

    captures = 0
    episode_steps: list[int] = []
    capture_steps: list[int] = []
    total_blocked = 0
    total_agent_steps = 0

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_steps = 0
        captured = False

        while True:
            if qnet is not None:
                actions = greedy_actions(qnet, obs, device)
            else:
                actions = random_actions(env, obs)

            next_obs, _, terminations, truncations, infos = env.step(actions)
            ep_steps += 1

            for agent, info in infos.items():
                if info.get("blocked_move", False):
                    total_blocked += 1
                total_agent_steps += 1
                if info.get("evaders_remaining", 1) == 0:
                    captured = True

            done = any(terminations.values()) or any(truncations.values()) or not env.agents
            obs = next_obs

            if done:
                break

        episode_steps.append(ep_steps)
        if captured:
            captures += 1
            capture_steps.append(ep_steps)

    env.close()

    n = args.episodes
    capture_rate = captures / n
    escape_rate = 1.0 - capture_rate
    avg_steps = float(np.mean(episode_steps))
    avg_steps_capture = float(np.mean(capture_steps)) if capture_steps else None
    collision_rate = total_blocked / max(1, total_agent_steps)

    metrics = {
        "map": args.map,
        "episodes": n,
        "seed": args.seed,
        "policy": "random" if args.random_policy else str(args.checkpoint),
        "capture_rate": round(capture_rate, 4),
        "escape_rate": round(escape_rate, 4),
        "avg_steps": round(avg_steps, 2),
        "avg_steps_capture": round(avg_steps_capture, 2) if avg_steps_capture else None,
        "collision_rate": round(collision_rate, 4),
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained pursuit agent")
    p.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint file")
    p.add_argument("--random-policy", action="store_true", help="Use random actions (no checkpoint needed)")
    p.add_argument("--map", default="easy_open", choices=["easy_open", "center_block", "split_barrier"])
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None, help="Path to save JSON results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.random_policy and not args.checkpoint:
        print("Error: provide --checkpoint or --random-policy")
        sys.exit(1)
    run_eval(args)
