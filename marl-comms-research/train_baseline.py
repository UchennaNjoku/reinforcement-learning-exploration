"""No-communication DQN baseline training script.

All three pursuers share one Q-network (parameter sharing).
Each step, every active agent contributes its transition to the shared buffer.

Usage:
    python train_baseline.py
    python train_baseline.py --map easy_open --episodes 2000 --seed 0
    python train_baseline.py --help
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Allow imports from the project root.
sys.path.insert(0, str(Path(__file__).parent))

from envs import make_fixed_pursuit_env
from models.qnet import PursuitQNet
from training.replay_buffer import ReplayBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AGENT_IDS: dict[str, int] = {
    "pursuer_0": 0,
    "pursuer_1": 1,
    "pursuer_2": 2,
}


def select_actions(
    qnet: PursuitQNet,
    observations: dict[str, np.ndarray],
    epsilon: float,
    n_actions: int,
    device: torch.device,
) -> dict[str, int]:
    """Epsilon-greedy action selection for all active agents."""
    actions: dict[str, int] = {}
    for agent, obs in observations.items():
        if random.random() < epsilon:
            actions[agent] = random.randrange(n_actions)
        else:
            obs_t = torch.from_numpy(obs.transpose(2, 0, 1)).unsqueeze(0).to(device)
            aid_t = torch.tensor([AGENT_IDS[agent]], dtype=torch.long, device=device)
            with torch.no_grad():
                q = qnet(obs_t, aid_t)
            actions[agent] = int(q.argmax(dim=1).item())
    return actions


def update_target(online: PursuitQNet, target: PursuitQNet) -> None:
    target.load_state_dict(online.state_dict())


def compute_loss(
    batch,
    online: PursuitQNet,
    target: PursuitQNet,
    gamma: float,
) -> torch.Tensor:
    q_values = online(batch.obs, batch.agent_ids)                       # (B, A)
    q_taken = q_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1) # (B,)

    with torch.no_grad():
        next_q = target(batch.next_obs, batch.agent_ids)                # (B, A)
        next_q_max = next_q.max(dim=1).values                          # (B,)
        td_target = batch.rewards + gamma * next_q_max * (1.0 - batch.dones)

    return F.mse_loss(q_taken, td_target)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = results_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    env = make_fixed_pursuit_env(
        map_name=args.map,
        n_catch=args.n_catch,
        distance_reward_scale=args.distance_reward_scale,
    )
    n_actions = env.action_space("pursuer_0").n

    online_net = PursuitQNet(n_agents=3, n_actions=n_actions).to(device)
    target_net = copy.deepcopy(online_net).to(device)
    target_net.eval()

    optimizer = torch.optim.Adam(online_net.parameters(), lr=args.lr)
    buffer = ReplayBuffer(capacity=args.buffer_size, device=str(device))

    # Epsilon schedule
    epsilon = args.eps_start
    eps_decay_per_step = (args.eps_start - args.eps_end) / max(1, args.eps_decay_steps)

    total_steps = 0
    episode_log: list[dict] = []

    print(f"Training for {args.episodes} episodes on map '{args.map}' | seed {args.seed}")
    print(f"Warmup steps: {args.warmup_steps} | batch size: {args.batch_size}")

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_rewards = {a: 0.0 for a in env.possible_agents}
        ep_steps = 0
        captured = False

        while True:
            actions = select_actions(online_net, obs, epsilon, n_actions, device)

            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            ep_steps += 1
            total_steps += 1

            done_global = (
                any(terminations.values()) or any(truncations.values())
                or not env.agents
            )

            # Store one transition per agent
            for agent in rewards:
                n_obs = next_obs.get(agent, obs[agent])  # agent may have left
                done_agent = terminations.get(agent, False) or truncations.get(agent, False)
                buffer.push(
                    obs=obs[agent],
                    agent_id=AGENT_IDS[agent],
                    action=actions[agent],
                    reward=rewards[agent],
                    next_obs=n_obs,
                    done=done_global,
                )
                ep_rewards[agent] += rewards[agent]

            obs = next_obs

            # Check capture
            for info in infos.values():
                if info.get("evaders_remaining", 1) == 0:
                    captured = True

            # Learning step
            if (
                total_steps >= args.warmup_steps
                and len(buffer) >= args.batch_size
            ):
                batch = buffer.sample(args.batch_size)
                loss = compute_loss(batch, online_net, target_net, args.gamma)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online_net.parameters(), 10.0)
                optimizer.step()

                if total_steps % args.target_update_freq == 0:
                    update_target(online_net, target_net)

            # Epsilon decay
            if total_steps >= args.warmup_steps:
                epsilon = max(args.eps_end, epsilon - eps_decay_per_step)

            if done_global:
                break

        team_reward = sum(ep_rewards.values()) / len(ep_rewards)
        episode_log.append({
            "episode": ep,
            "steps": ep_steps,
            "team_reward": round(team_reward, 4),
            "captured": captured,
            "epsilon": round(epsilon, 4),
            "total_steps": total_steps,
        })

        if ep % args.log_interval == 0:
            recent = episode_log[-args.log_interval:]
            cap_rate = sum(r["captured"] for r in recent) / len(recent)
            avg_steps = sum(r["steps"] for r in recent) / len(recent)
            avg_reward = sum(r["team_reward"] for r in recent) / len(recent)
            print(
                f"Ep {ep:5d}/{args.episodes} | "
                f"cap {cap_rate:.2f} | "
                f"steps {avg_steps:5.1f} | "
                f"reward {avg_reward:+.3f} | "
                f"eps {epsilon:.3f} | "
                f"buf {len(buffer):,}"
            )

        if ep % args.save_interval == 0:
            ckpt_path = ckpt_dir / f"baseline_ep{ep:06d}.pt"
            torch.save(
                {
                    "episode": ep,
                    "total_steps": total_steps,
                    "model_state": online_net.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epsilon": epsilon,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  Checkpoint saved → {ckpt_path}")

    env.close()

    # Save final model
    final_path = ckpt_dir / "baseline_final.pt"
    torch.save(
        {
            "episode": args.episodes,
            "total_steps": total_steps,
            "model_state": online_net.state_dict(),
            "args": vars(args),
        },
        final_path,
    )
    print(f"Final model saved → {final_path}")

    # Save episode log
    log_path = results_dir / f"baseline_train_log_seed{args.seed}.json"
    with open(log_path, "w") as f:
        json.dump(episode_log, f, indent=2)
    print(f"Training log saved → {log_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="No-communication DQN baseline training")
    p.add_argument("--map", default="easy_open", choices=["easy_open", "center_block", "split_barrier"])
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--warmup-steps", type=int, default=1_000)
    p.add_argument("--target-update-freq", type=int, default=500)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-steps", type=int, default=50_000)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--n-catch", type=int, default=1, help="Pursuers needed adjacent to catch (1=easy, 2=hard)")
    p.add_argument("--distance-reward-scale", type=float, default=0.1, help="Scale for dense distance-to-prey reward (0=off)")
    p.add_argument("--results-dir", default="results/baseline")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
