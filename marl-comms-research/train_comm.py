"""Communication-enabled DQN training script.

Each agent independently selects a move (via move Q-head) and a message
(via message Q-head) at every step. Both heads share a CNN trunk and are
trained with the same TD target. Received messages from teammates at t-1
are part of each agent's state at t.

All agents share one CommQNet (parameter sharing).

Usage:
    python train_comm.py --vocab-size 4   # 2-bit comm
    python train_comm.py --vocab-size 16  # 4-bit comm
    python train_comm.py --help
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from envs import make_fixed_pursuit_env
from models.comm_qnet import CommQNet
from training.comm_replay_buffer import CommReplayBuffer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_IDS: dict[str, int] = {
    "pursuer_0": 0,
    "pursuer_1": 1,
    "pursuer_2": 2,
}
N_AGENTS = 3


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------

def zero_messages(vocab_size: int) -> dict[str, np.ndarray]:
    """Zero message state at episode start."""
    return {agent: np.zeros(vocab_size, dtype=np.float32) for agent in AGENT_IDS}


def msg_to_onehot(msg_idx: int, vocab_size: int) -> np.ndarray:
    v = np.zeros(vocab_size, dtype=np.float32)
    v[msg_idx] = 1.0
    return v


def build_received(agent: str, prev_messages: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate all other agents' previous messages for `agent`."""
    others = [m for a, m in sorted(prev_messages.items()) if a != agent]
    return np.concatenate(others, axis=0)


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------

def select_actions(
    qnet: CommQNet,
    observations: dict[str, np.ndarray],
    prev_messages: dict[str, np.ndarray],
    epsilon: float,
    device: torch.device,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Epsilon-greedy selection of moves and messages independently.

    Returns:
        joint_actions : {agent: move * vocab_size + msg}  (stored in buffer)
        move_actions  : {agent: move_idx}                 (sent to env)
        msg_actions   : {agent: msg_idx}                  (sent to teammates)
    """
    joint_actions: dict[str, int] = {}
    move_actions:  dict[str, int] = {}
    msg_actions:   dict[str, int] = {}

    for agent, obs in observations.items():
        rand_move = random.random() < epsilon
        rand_msg  = random.random() < epsilon

        if rand_move and rand_msg:
            move = random.randrange(qnet.n_move_actions)
            msg  = random.randrange(qnet.vocab_size)
        else:
            obs_t  = torch.from_numpy(obs.transpose(2, 0, 1)).unsqueeze(0).to(device)
            aid_t  = torch.tensor([AGENT_IDS[agent]], dtype=torch.long, device=device)
            recv_t = torch.from_numpy(
                build_received(agent, prev_messages)
            ).unsqueeze(0).to(device)
            with torch.no_grad():
                move_q, msg_q = qnet(obs_t, aid_t, recv_t)
            move = random.randrange(qnet.n_move_actions) if rand_move else int(move_q.argmax(dim=1).item())
            msg  = random.randrange(qnet.vocab_size)     if rand_msg  else int(msg_q.argmax(dim=1).item())

        joint_actions[agent] = move * qnet.vocab_size + msg
        move_actions[agent]  = move
        msg_actions[agent]   = msg

    return joint_actions, move_actions, msg_actions


def update_target(online: CommQNet, target: CommQNet) -> None:
    target.load_state_dict(online.state_dict())


def compute_loss(
    batch,
    online: CommQNet,
    target: CommQNet,
    gamma: float,
) -> torch.Tensor:
    """DQN loss for move head + message head, sharing one TD target.

    TD target uses the move head's max Q at the next state — moves drive
    the team reward, so the move head anchors the value estimate for both.
    """
    vocab = online.vocab_size

    # Decode stored joint actions into moves and messages
    move_actions = batch.actions // vocab
    msg_actions  = batch.actions  % vocab

    move_q_online, msg_q_online = online(batch.obs, batch.agent_ids, batch.received_msgs)

    move_q_taken = move_q_online.gather(1, move_actions.unsqueeze(1)).squeeze(1)
    msg_q_taken  = msg_q_online.gather(1,  msg_actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_move_q, _ = target(batch.next_obs, batch.agent_ids, batch.next_received_msgs)
        td_target = batch.rewards + gamma * next_move_q.max(dim=1).values * (1.0 - batch.dones)

    move_loss = F.mse_loss(move_q_taken, td_target)
    msg_loss  = F.mse_loss(msg_q_taken,  td_target)
    return move_loss + msg_loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | vocab_size: {args.vocab_size}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = results_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    env = make_fixed_pursuit_env(
        map_name=args.map,
        n_catch=args.n_catch,
        surround=False,
        distance_reward_scale=args.distance_reward_scale,
    )
    n_move_actions = env.action_space("pursuer_0").n

    online_net = CommQNet(
        n_agents=N_AGENTS,
        n_move_actions=n_move_actions,
        vocab_size=args.vocab_size,
    ).to(device)
    target_net = copy.deepcopy(online_net).to(device)
    target_net.eval()

    optimizer = torch.optim.Adam(online_net.parameters(), lr=args.lr)

    msg_dim = (N_AGENTS - 1) * args.vocab_size
    buffer = CommReplayBuffer(
        capacity=args.buffer_size,
        msg_dim=msg_dim,
        device=str(device),
    )

    epsilon = args.eps_start
    eps_decay_per_step = (args.eps_start - args.eps_end) / max(1, args.eps_decay_steps)

    total_steps = 0
    episode_log: list[dict] = []
    msg_log_buffer: list[dict] = []

    print(
        f"Training for {args.episodes} episodes on map '{args.map}' | seed {args.seed}\n"
        f"Warmup steps: {args.warmup_steps} | batch size: {args.batch_size}"
    )

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + ep)
        prev_messages = zero_messages(args.vocab_size)
        ep_rewards = {a: 0.0 for a in env.possible_agents}
        ep_steps   = 0
        captured   = False
        msg_log: list[dict[str, int]] = []

        while True:
            joint_actions, move_actions, msg_actions = select_actions(
                online_net, obs, prev_messages, epsilon, device
            )

            recv_now = {
                agent: build_received(agent, prev_messages)
                for agent in obs
            }

            next_obs, rewards, terminations, truncations, infos = env.step(move_actions)
            ep_steps   += 1
            total_steps += 1

            done_global = (
                any(terminations.values()) or any(truncations.values()) or not env.agents
            )

            new_messages = {
                agent: msg_to_onehot(msg_actions[agent], args.vocab_size)
                for agent in joint_actions
            }
            recv_next = {
                agent: build_received(agent, new_messages)
                for agent in obs
            }

            for agent in rewards:
                n_ob = next_obs.get(agent, obs[agent])
                buffer.push(
                    obs=obs[agent],
                    agent_id=AGENT_IDS[agent],
                    received_msgs=recv_now[agent],
                    action=joint_actions[agent],
                    reward=rewards[agent],
                    next_obs=n_ob,
                    next_received_msgs=recv_next[agent],
                    done=done_global,
                )
                ep_rewards[agent] += rewards[agent]

            msg_log.append({a: int(msg_actions.get(a, -1)) for a in AGENT_IDS})
            prev_messages = new_messages
            obs = next_obs

            for info in infos.values():
                if info.get("evaders_remaining", 1) == 0:
                    captured = True

            if total_steps >= args.warmup_steps and len(buffer) >= args.batch_size:
                batch = buffer.sample(args.batch_size)
                loss  = compute_loss(batch, online_net, target_net, args.gamma)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online_net.parameters(), 10.0)
                optimizer.step()

                if total_steps % args.target_update_freq == 0:
                    update_target(online_net, target_net)

            if total_steps >= args.warmup_steps:
                epsilon = max(args.eps_end, epsilon - eps_decay_per_step)

            if done_global:
                break

        msg_log_buffer.append({
            "episode":  ep,
            "captured": captured,
            "steps":    ep_steps,
            "messages": msg_log,
        })
        if len(msg_log_buffer) > args.msg_log_window:
            msg_log_buffer.pop(0)

        team_reward = sum(ep_rewards.values()) / len(ep_rewards)
        episode_log.append({
            "episode":     ep,
            "steps":       ep_steps,
            "team_reward": round(team_reward, 4),
            "captured":    captured,
            "epsilon":     round(epsilon, 4),
            "total_steps": total_steps,
        })

        if ep % args.log_interval == 0:
            recent    = episode_log[-args.log_interval:]
            cap_rate  = sum(r["captured"]    for r in recent) / len(recent)
            avg_steps = sum(r["steps"]       for r in recent) / len(recent)
            avg_rew   = sum(r["team_reward"] for r in recent) / len(recent)
            print(
                f"Ep {ep:5d}/{args.episodes} | "
                f"cap {cap_rate:.2f} | "
                f"steps {avg_steps:5.1f} | "
                f"reward {avg_rew:+.3f} | "
                f"eps {epsilon:.3f} | "
                f"buf {len(buffer):,}"
            )

        if ep % args.save_interval == 0:
            ckpt_path = ckpt_dir / f"comm{args.vocab_size}_ep{ep:06d}.pt"
            torch.save({
                "episode":         ep,
                "total_steps":     total_steps,
                "model_state":     online_net.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epsilon":         epsilon,
                "args":            vars(args),
            }, ckpt_path)
            print(f"  Checkpoint saved → {ckpt_path}")

    env.close()

    final_path = ckpt_dir / f"comm{args.vocab_size}_final.pt"
    torch.save({
        "episode":     args.episodes,
        "total_steps": total_steps,
        "model_state": online_net.state_dict(),
        "args":        vars(args),
    }, final_path)
    print(f"Final model saved → {final_path}")

    log_path = results_dir / f"comm{args.vocab_size}_train_log_seed{args.seed}.json"
    with open(log_path, "w") as f:
        json.dump(episode_log, f, indent=2)
    print(f"Training log saved → {log_path}")

    msg_log_path = results_dir / f"comm{args.vocab_size}_msg_log_seed{args.seed}.json"
    with open(msg_log_path, "w") as f:
        json.dump(msg_log_buffer, f, indent=2)
    print(f"Message log saved  → {msg_log_path}  ({len(msg_log_buffer)} episodes)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Communicating DQN training")
    p.add_argument("--map",                   default="easy_open",
                   choices=["easy_open", "center_block", "split_barrier"])
    p.add_argument("--vocab-size",            type=int,   default=4,
                   help="Message vocabulary size (4=2-bit, 16=4-bit)")
    p.add_argument("--episodes",              type=int,   default=5000)
    p.add_argument("--seed",                  type=int,   default=0)
    p.add_argument("--lr",                    type=float, default=1e-3)
    p.add_argument("--gamma",                 type=float, default=0.99)
    p.add_argument("--batch-size",            type=int,   default=64)
    p.add_argument("--buffer-size",           type=int,   default=100_000)
    p.add_argument("--warmup-steps",          type=int,   default=5_000)
    p.add_argument("--target-update-freq",    type=int,   default=500)
    p.add_argument("--eps-start",             type=float, default=1.0)
    p.add_argument("--eps-end",               type=float, default=0.05)
    p.add_argument("--eps-decay-steps",       type=int,   default=500_000)
    p.add_argument("--n-catch",               type=int,   default=1)
    p.add_argument("--distance-reward-scale", type=float, default=0.1)
    p.add_argument("--msg-log-window",        type=int,   default=500)
    p.add_argument("--log-interval",          type=int,   default=100)
    p.add_argument("--save-interval",         type=int,   default=500)
    p.add_argument("--results-dir",           default="results/comm")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
