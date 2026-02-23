"""
DQN Training for GridWorld Navigation
=======================================
Combined best-practices implementation with optional curriculum learning.

Features:
    - Double DQN (online selects action, target evaluates)
    - 3-channel spatial grid + relative scalar features
    - Potential-based reward shaping (with per-phase weight control)
    - BFS distance map for wall-aware shaping (computed once per episode)
    - Linear epsilon schedule (predictable exploration)
    - Experience replay with configurable warmup
    - Best-model tracking by evaluation success rate
    - Checkpoint resume support
    - Curriculum learning (--curriculum flag)

Usage:
    python train.py                              # Single-phase training
    python train.py --curriculum                 # Curriculum (progressive difficulty)
    python train.py --curriculum --preset aggressive  # Faster curriculum
    python train.py --episodes 10000             # More episodes (single-phase)
    python train.py --no-reward-shaping          # Sparse reward only
    python train.py --resume checkpoints/best_model.pt

Author: Chenna (CS Senior, Bethune-Cookman University)
"""

from __future__ import annotations

import argparse
import os
import time
import random
from collections import deque
from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("PyTorch is required. Install with: pip install torch")
    exit(1)

from env import GridWorldEnv
from config import ENV_CONFIG, AGENT_CONFIG, TRAIN_CONFIG, PATHS


# ─────────────────────────────────────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-size circular buffer for experience replay."""

    def __init__(self, capacity: int):
        self.buf: deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition):
        self.buf.append(t)

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s = np.stack([b.s for b in batch])
        a = np.array([b.a for b in batch], dtype=np.int64)
        r = np.array([b.r for b in batch], dtype=np.float32)
        s2 = np.stack([b.s2 for b in batch])
        done = np.array([b.done for b in batch], dtype=np.float32)
        return s, a, r, s2, done

    def __len__(self) -> int:
        return len(self.buf)


# ─────────────────────────────────────────────────────────────────────────────
# State Encoder
# ─────────────────────────────────────────────────────────────────────────────
class StateEncoder:
    """
    Encodes GridWorld observation into a flat feature vector.

    Spatial channels (3 × H × W):
        ch0: walls (border + obstacle)
        ch1: goal position
        ch2: agent position

    Scalar features (7):
        agent (x,y), goal (x,y), relative (dx,dy), manhattan distance
        — all normalized to [0, 1]

    Total dim = 3 * size * size + 7
    """

    def __init__(self, grid_size: int):
        self.size = grid_size
        self.feature_dim = 3 * grid_size * grid_size + 7

    def encode(self, obs: dict, env: GridWorldEnv) -> np.ndarray:
        size = self.size

        # Spatial channels
        grid = env.grid
        ch_walls = (grid == GridWorldEnv.TILE_WALL).astype(np.float32)
        ch_goal = np.zeros((size, size), dtype=np.float32)
        ch_agent = np.zeros((size, size), dtype=np.float32)

        gx, gy = int(obs["goal_pos"][0]), int(obs["goal_pos"][1])
        ax, ay = int(obs["agent_pos"][0]), int(obs["agent_pos"][1])
        ch_goal[gy, gx] = 1.0
        ch_agent[ay, ax] = 1.0

        spatial = np.stack([ch_walls, ch_goal, ch_agent]).reshape(-1)

        # Scalar features
        norm = size - 1
        scalars = np.array([
            ax / norm, ay / norm,
            gx / norm, gy / norm,
            (gx - ax) / norm, (gy - ay) / norm,
            (abs(gx - ax) + abs(gy - ay)) / (2 * (size - 2)),
        ], dtype=np.float32)

        return np.concatenate([spatial, scalars])

    def encode_tensor(self, obs: dict, env: GridWorldEnv, device: torch.device) -> torch.Tensor:
        return torch.tensor(self.encode(obs, env), dtype=torch.float32, device=device).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# Q-Network
# ─────────────────────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    """MLP: 512 → 256 → 128 → n_actions"""

    def __init__(self, input_dim: int, n_actions: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# BFS Distance Map
# ─────────────────────────────────────────────────────────────────────────────
def bfs_distance_map(env: GridWorldEnv, goal_xy, size: int) -> np.ndarray:
    """
    Reverse BFS from goal. Returns dist[y, x] = shortest path steps
    from (x, y) to goal, respecting walls. Computed once per episode.
    """
    gx, gy = int(goal_xy[0]), int(goal_xy[1])
    INF = 10**9
    dist = np.full((size, size), INF, dtype=np.int32)

    q = deque()
    dist[gy, gx] = 0
    q.append((gx, gy))

    while q:
        x, y = q.popleft()
        d = dist[y, x]
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if env._is_blocked(nx, ny):
                continue
            if dist[ny, nx] > d + 1:
                dist[ny, nx] = d + 1
                q.append((nx, ny))

    return dist


# ─────────────────────────────────────────────────────────────────────────────
# Reward Shaping
# ─────────────────────────────────────────────────────────────────────────────
def shaped_reward(
    obs: dict,
    prev_obs: dict,
    terminated: bool,
    truncated: bool,
    dist_map: np.ndarray | None,
    grid_size: int,
    shaping_weight: float = 1.0,
) -> float:
    """
    Reward function with controllable BFS-distance shaping.

        +5.0    goal reached
        -0.01   step penalty
        ±scale  BFS distance delta, normalized by max possible distance
        -0.03   wall-bump penalty (agent didn't move)
        -0.5    truncation penalty (ran out of steps)

    Key fix: normalize by max_dist (constant per grid size), NOT by prev_d.
    This gives consistent reward scale regardless of agent position.
    Dividing by prev_d caused huge variance — close to goal gave ~0.5 per step,
    far from goal gave ~0.07 per step. Network couldn't learn stable Q-values.
    """
    if terminated:
        return 5.0

    # Step penalty — small but present
    reward = -0.01

    # BFS-distance shaping (only if weight > 0 and map available)
    if shaping_weight > 0.0 and dist_map is not None:
        pax, pay = int(prev_obs["agent_pos"][0]), int(prev_obs["agent_pos"][1])
        cax, cay = int(obs["agent_pos"][0]), int(obs["agent_pos"][1])

        prev_d = int(dist_map[pay, pax])
        curr_d = int(dist_map[cay, cax])

        # Normalize by max possible distance (constant for grid size)
        # On 12×12 grid: max_dist = 2*(12-2) = 20
        max_dist = 2 * (grid_size - 2)
        reward += shaping_weight * (prev_d - curr_d) / max_dist

    # Wall-bump / no-move penalty
    if (obs["agent_pos"][0] == prev_obs["agent_pos"][0]) and \
       (obs["agent_pos"][1] == prev_obs["agent_pos"][1]):
        reward -= 0.03

    # Truncation penalty
    if truncated:
        reward -= 0.5

    return reward


# ─────────────────────────────────────────────────────────────────────────────
# Epsilon Schedule
# ─────────────────────────────────────────────────────────────────────────────
def linear_epsilon(episode: int, start: float, end: float, decay_episodes: int) -> float:
    """Linear decay from start → end over decay_episodes, then constant."""
    if episode >= decay_episodes:
        return end
    return start + (end - start) * (episode / max(1, decay_episodes))


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    q_net: nn.Module,
    encoder: StateEncoder,
    device: torch.device,
    n_episodes: int = 25,
    grid_size: int = 12,
    max_steps: int = 120,
    wall_length: int = 4,
    render: bool = False,
) -> dict:
    """Greedy evaluation. Returns success_rate, avg_steps, min/max steps."""
    env = GridWorldEnv(
        size=grid_size, max_steps=max_steps, wall_length=wall_length,
        render_mode="human" if render else None,
    )
    q_net.eval()
    successes = 0
    steps_list = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, steps = False, 0
        while not done:
            state = encoder.encode_tensor(obs, env, device)
            action = int(q_net(state).argmax(dim=1).item())
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            if terminated:
                successes += 1
            if render:
                time.sleep(0.05)
        steps_list.append(steps)

    q_net.train()
    env.close()
    return {
        "success_rate": successes / n_episodes,
        "avg_steps": float(np.mean(steps_list)),
        "min_steps": min(steps_list),
        "max_steps": max(steps_list),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Save / Load
# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(filepath, episode, q_net, target_net, optimizer,
                    epsilon, global_step, best_success_rate, extra=None):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    data = {
        "episode": episode,
        "q_state_dict": q_net.state_dict(),
        "target_state_dict": target_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epsilon": epsilon,
        "global_step": global_step,
        "best_success_rate": best_success_rate,
    }
    if extra:
        data.update(extra)
    torch.save(data, filepath)


def load_checkpoint(filepath, q_net, target_net, optimizer, device):
    ckpt = torch.load(filepath, map_location=device, weights_only=True)
    q_net.load_state_dict(ckpt["q_state_dict"])
    target_net.load_state_dict(ckpt["target_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Core Training Phase
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrainState:
    """Mutable state carried across curriculum phases."""
    q_net: nn.Module
    target_net: nn.Module
    optimizer: optim.Optimizer
    replay: ReplayBuffer
    encoder: StateEncoder
    device: torch.device
    global_step: int = 0
    best_success_rate: float = 0.0


def train_phase(
    state: TrainState,
    *,
    # Environment
    grid_size: int = 12,
    max_steps: int = 120,
    wall_length: int = 4,
    # Training
    n_episodes: int = 5000,
    use_reward_shaping: bool = True,
    shaping_weight: float = 1.0,
    # Epsilon
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_episodes: int = 12_000,
    # DQN
    gamma: float = 0.99,
    batch_size: int = 256,
    target_update_steps: int = 1000,
    grad_clip: float = 10.0,
    # Logging
    log_interval: int = 50,
    eval_interval: int = 500,
    eval_episodes: int = 25,
    save_interval: int = 1000,
    save_dir: str = "./checkpoints",
    render_eval: bool = False,
    # Curriculum (optional)
    phase_name: str = "",
    on_eval: object = None,     # callable(success_rate) -> bool (should_stop)
) -> dict:
    """
    Run one training phase. Returns summary dict.

    If on_eval is provided, it's called after each evaluation with the
    success rate. If it returns True, the phase ends early (threshold met).
    """
    env = GridWorldEnv(size=grid_size, max_steps=max_steps, wall_length=wall_length, render_mode=None)

    q_net = state.q_net
    target_net = state.target_net
    optimizer = state.optimizer
    replay = state.replay
    encoder = state.encoder
    device = state.device

    recent_rewards = deque(maxlen=100)
    recent_successes = deque(maxlen=100)
    recent_losses = deque(maxlen=500)
    start_time = time.time()
    os.makedirs(save_dir, exist_ok=True)

    # Precompute whether we need BFS maps this phase
    need_dist_map = use_reward_shaping and shaping_weight > 0.0

    phase_best_sr = 0.0
    total_ep = 0

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        prev_obs = obs
        s = encoder.encode(obs, env)

        # BFS distance map: computed ONCE per episode (walls & goal are fixed)
        dist_map = bfs_distance_map(env, obs["goal_pos"], grid_size) if need_dist_map else None

        ep_reward, steps, done, reached_goal = 0.0, 0, False, False
        eps = linear_epsilon(ep, eps_start, eps_end, eps_decay_episodes)

        while not done:
            state.global_step += 1
            steps += 1

            # ε-greedy action
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                    action = int(q_net(st).argmax(dim=1).item())

            next_obs, base_reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if terminated:
                reached_goal = True

            # Compute reward — dist_map lookup is O(1)
            if use_reward_shaping:
                reward = shaped_reward(
                    next_obs, prev_obs,
                    terminated, truncated,
                    dist_map, grid_size,
                    shaping_weight=shaping_weight,
                )
            else:
                reward = float(base_reward)

            s2 = encoder.encode(next_obs, env)
            replay.push(Transition(s, action, reward, s2, done))

            # ── Double DQN update ──
            if len(replay) >= batch_size:
                bs, ba, br, bs2, bdone = replay.sample(batch_size)
                bs_t = torch.tensor(bs, dtype=torch.float32, device=device)
                ba_t = torch.tensor(ba, dtype=torch.long, device=device).unsqueeze(1)
                br_t = torch.tensor(br, dtype=torch.float32, device=device)
                bs2_t = torch.tensor(bs2, dtype=torch.float32, device=device)
                bdone_t = torch.tensor(bdone, dtype=torch.float32, device=device)

                q_vals = q_net(bs_t).gather(1, ba_t).squeeze(1)
                with torch.no_grad():
                    next_a = q_net(bs2_t).argmax(dim=1, keepdim=True)
                    next_q = target_net(bs2_t).gather(1, next_a).squeeze(1)
                    target_q = br_t + (1.0 - bdone_t) * gamma * next_q

                loss = nn.SmoothL1Loss()(q_vals, target_q)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), grad_clip)
                optimizer.step()
                recent_losses.append(loss.item())

                if state.global_step % target_update_steps == 0:
                    target_net.load_state_dict(q_net.state_dict())

            s = s2
            prev_obs = next_obs
            ep_reward += reward

        recent_rewards.append(ep_reward)
        recent_successes.append(1.0 if reached_goal else 0.0)
        total_ep = ep

        # ── Logging ──
        if ep % log_interval == 0:
            avg_r = np.mean(recent_rewards)
            avg_succ = np.mean(recent_successes) * 100
            avg_loss = np.mean(recent_losses) if recent_losses else 0.0
            tag = f"[{phase_name}] " if phase_name else ""
            print(
                f"  {tag}Ep {ep:>5d}/{n_episodes} │ "
                f"R={avg_r:>7.2f} │ "
                f"Succ={avg_succ:>5.1f}% │ "
                f"ε={eps:.4f} │ "
                f"Loss={avg_loss:.4f} │ "
                f"buf={len(replay):>6d}"
            )

        # ── Evaluation ──
        should_stop = False
        if eval_interval > 0 and ep % eval_interval == 0:
            ev = evaluate(
                q_net, encoder, device,
                n_episodes=eval_episodes,
                grid_size=grid_size,
                max_steps=max_steps,
                wall_length=wall_length,
                render=render_eval,
            )
            sr = ev["success_rate"]
            phase_best_sr = max(phase_best_sr, sr)
            tag = f"[{phase_name}] " if phase_name else ""
            print(
                f"\n  {tag}── EVAL @Ep {ep} ──  "
                f"Success={sr*100:.1f}% │ "
                f"AvgSteps={ev['avg_steps']:.1f} │ "
                f"Range=[{ev['min_steps']}, {ev['max_steps']}]"
            )

            # Best model tracking — only save best when evaluating at target difficulty
            # (wall_length >= 3), otherwise early easy-phase evals dominate
            if sr > state.best_success_rate and wall_length >= 3:
                state.best_success_rate = sr
                save_checkpoint(
                    os.path.join(save_dir, "best_model.pt"),
                    ep, q_net, target_net, optimizer, eps,
                    state.global_step, state.best_success_rate,
                    extra={"phase": phase_name, "wall_length": wall_length},
                )
                print(f"  ★ New global best: {sr*100:.1f}% → saved best_model.pt")

            if wall_length == 4 and sr >= phase_best_sr:
                save_checkpoint(
                    os.path.join(save_dir, "best_wall4.pt"),
                    ep, q_net, target_net, optimizer, eps,
                    state.global_step, state.best_success_rate,
                    extra={"phase": phase_name, "wall_length": wall_length},
                )

            # Curriculum callback
            if on_eval is not None:
                should_stop = on_eval(sr)
                if should_stop:
                    print(f"  ✓ Phase advancement threshold met!")

            print()

        # ── Periodic checkpoint ──
        if save_interval > 0 and ep % save_interval == 0:
            save_checkpoint(
                os.path.join(save_dir, f"checkpoint_step{state.global_step}.pt"),
                ep, q_net, target_net, optimizer, eps,
                state.global_step, state.best_success_rate,
                extra={"phase": phase_name, "wall_length": wall_length},
            )

        if should_stop:
            break

    env.close()
    elapsed = time.time() - start_time

    return {
        "episodes": total_ep,
        "phase_best_sr": phase_best_sr,
        "global_best_sr": state.best_success_rate,
        "global_step": state.global_step,
        "elapsed": elapsed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Warmup
# ─────────────────────────────────────────────────────────────────────────────
def warmup_buffer(replay, encoder, grid_size, max_steps, wall_length, n_steps, seed=42, policy="random", eps=0.2):
    """
    Fill replay buffer.
    policy:
      - "random": pure random actions
      - "bfs": mostly actions that reduce BFS distance (with eps random)
    """
    env = GridWorldEnv(size=grid_size, max_steps=max_steps, wall_length=wall_length, render_mode=None)
    obs, _ = env.reset(seed=seed)

    dist_map = None
    if policy == "bfs":
        dist_map = bfs_distance_map(env, obs["goal_pos"], grid_size)

    for _ in range(n_steps):
        if policy == "bfs" and dist_map is not None and random.random() > eps:
            ax, ay = map(int, obs["agent_pos"])
            best_a, best_d = None, 10**9

            for a in range(env.action_space.n):
                nx, ny = ax, ay
                if a == 0: ny -= 1
                elif a == 1: ny += 1
                elif a == 2: nx -= 1
                elif a == 3: nx += 1

                # block check (stay put if blocked)
                if env._is_blocked(nx, ny):
                    nx, ny = ax, ay

                d = int(dist_map[ny, nx])
                if d < best_d:
                    best_d, best_a = d, a

            action = best_a if best_a is not None else env.action_space.sample()
        else:
            action = env.action_space.sample()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        s  = encoder.encode(obs, env)
        s2 = encoder.encode(next_obs, env)
        replay.push(Transition(s, action, float(reward), s2, done))

        obs = next_obs
        if done:
            obs, _ = env.reset()
            if policy == "bfs":
                dist_map = bfs_distance_map(env, obs["goal_pos"], grid_size)


# ─────────────────────────────────────────────────────────────────────────────
# Single-Phase Training
# ─────────────────────────────────────────────────────────────────────────────
def train_single(args: argparse.Namespace):
    """Standard single-phase DQN training."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    encoder = StateEncoder(args.size)

    q_net = QNetwork(encoder.feature_dim, 4).to(device)
    target_net = QNetwork(encoder.feature_dim, 4).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.replay_size)

    if args.resume and os.path.exists(args.resume):
        load_checkpoint(args.resume, q_net, target_net, optimizer, device)
        print(f"  Resumed from {args.resume}")

    ts = TrainState(q_net=q_net, target_net=target_net, optimizer=optimizer,
                    replay=replay, encoder=encoder, device=device)

    param_count = sum(p.numel() for p in q_net.parameters())
    print("=" * 70)
    print("  DQN TRAINING — GridWorld Navigation (Single Phase)")
    print("=" * 70)
    print(f"  Device:            {device}")
    print(f"  State dim:         {encoder.feature_dim}")
    print(f"  Network params:    {param_count:,}")
    print(f"  Grid:              {args.size}×{args.size} ({args.size-2}×{args.size-2} playable)")
    print(f"  Wall length:       {args.wall_length}")
    print(f"  Episodes:          {args.episodes}")
    print(f"  Reward shaping:    {'ON' if not args.no_reward_shaping else 'OFF'}")
    print(f"  Shaping weight:    {args.shaping_weight}")
    print(f"  Learning rate:     {args.lr}")
    print(f"  Double DQN:        ON")
    print(f"  Epsilon:           {args.eps_start} → {args.eps_end} over {args.eps_decay_episodes} eps")
    print("=" * 70 + "\n")

    print(f"  Warming up ({args.warmup_steps} random steps)...", end=" ", flush=True)
    warmup_buffer(replay, encoder, args.size, args.max_steps, args.wall_length,
                  args.warmup_steps, args.seed)
    print(f"done ({len(replay)} transitions)\n")

    result = train_phase(
        ts,
        grid_size=args.size, max_steps=args.max_steps, wall_length=args.wall_length,
        n_episodes=args.episodes, use_reward_shaping=not args.no_reward_shaping,
        shaping_weight=args.shaping_weight,
        eps_start=args.eps_start, eps_end=args.eps_end,
        eps_decay_episodes=args.eps_decay_episodes,
        gamma=args.gamma, batch_size=args.batch_size,
        target_update_steps=args.target_update_steps, grad_clip=args.grad_clip,
        log_interval=args.log_interval, eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes, save_interval=args.save_interval,
        save_dir=args.save_dir, render_eval=args.render_eval,
    )

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Episodes:     {result['episodes']}")
    print(f"  Global steps: {result['global_step']:,}")
    print(f"  Best SR:      {result['global_best_sr']*100:.1f}%")
    print(f"  Time:         {result['elapsed']:.1f}s ({result['elapsed']/60:.1f} min)")

    print(f"\n  ── Final Evaluation (50 episodes, greedy) ──")
    final = evaluate(q_net, encoder, device, n_episodes=50,
                     grid_size=args.size, max_steps=args.max_steps,
                     wall_length=args.wall_length)
    print(f"  Success={final['success_rate']*100:.1f}% │ "
          f"AvgSteps={final['avg_steps']:.1f} │ "
          f"Range=[{final['min_steps']}, {final['max_steps']}]")

    save_checkpoint(os.path.join(args.save_dir, "final_model.pt"),
                    result['episodes'], q_net, target_net, optimizer, args.eps_end,
                    ts.global_step, ts.best_success_rate)
    print(f"  [Final model → final_model.pt]\nDone! 🎯")


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum Training
# ─────────────────────────────────────────────────────────────────────────────
def train_curriculum(args: argparse.Namespace):
    """Multi-phase curriculum training with progressive difficulty."""
    from curriculum import default_curriculum, aggressive_curriculum, PhaseTracker

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    encoder = StateEncoder(args.size)

    q_net = QNetwork(encoder.feature_dim, 4).to(device)
    target_net = QNetwork(encoder.feature_dim, 4).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.replay_size)

    if args.resume and os.path.exists(args.resume):
        load_checkpoint(args.resume, q_net, target_net, optimizer, device)
        print(f"  Resumed weights from {args.resume}")

    ts = TrainState(q_net=q_net, target_net=target_net, optimizer=optimizer,
                    replay=replay, encoder=encoder, device=device)

    # Select curriculum preset
    curriculum = aggressive_curriculum() if args.preset == "aggressive" else default_curriculum()
    curriculum.eval_episodes = args.eval_episodes
    curriculum.eval_interval = args.eval_interval

    param_count = sum(p.numel() for p in q_net.parameters())
    print("=" * 70)
    print("  DQN TRAINING — GridWorld Navigation (CURRICULUM)")
    print("=" * 70)
    print(f"  Device:            {device}")
    print(f"  State dim:         {encoder.feature_dim}")
    print(f"  Network params:    {param_count:,}")
    print(f"  Grid:              {args.size}×{args.size}")
    print(f"  Preset:            {args.preset}")
    print(f"  Reward shaping:    {'ON (per-phase weight)' if not args.no_reward_shaping else 'OFF'}")
    print(f"  Learning rate:     {args.lr}")
    print(f"  Double DQN:        ON")
    print(f"  Phases:            {len(curriculum.phases)}")
    print(f"  Max total eps:     {curriculum.total_max_episodes}")
    print()
    print(curriculum.summary())
    print("=" * 70 + "\n")


    total_start = time.time()
    total_episodes = 0

    for phase_idx, phase in enumerate(curriculum.phases, 1):
        
        ts.replay = ReplayBuffer(args.replay_size)

        warm_steps = args.warmup_steps if phase_idx == 1 else args.warmup_steps * 5

        # phase-specific warmup
        warmup_buffer(
            ts.replay,
            ts.encoder,
            args.size,
            args.max_steps,
            wall_length=phase.wall_length,
            n_steps=warm_steps,
            seed=args.seed + phase_idx,
            policy="bfs", 
            eps=0.05
        )

        tracker = PhaseTracker(phase=phase)

        print("─" * 70)
        print(f"  PHASE {phase_idx}/{len(curriculum.phases)}: {phase.name}")
        print(f"  wall_length={phase.wall_length}  │  "
              f"max_eps={phase.max_episodes}  │  "
              f"advance@{phase.advance_threshold*100:.0f}% "
              f"(×{phase.consecutive_required})  │  "
              f"ε={phase.eps_reset:.2f}  │  "
              f"shaping={phase.shaping_weight:.1f}")
        print("─" * 70 + "\n")

        # Epsilon decays over 85% of this phase's budget (slower decay)
        phase_decay = int(phase.max_episodes * 0.85)

        def make_eval_callback(trk):
            """Create callback that checks phase advancement."""
            def on_eval(sr: float) -> bool:
                return trk.record_eval(sr)
            return on_eval

        result = train_phase(
            ts,
            grid_size=args.size, max_steps=args.max_steps,
            wall_length=phase.wall_length,
            n_episodes=phase.max_episodes,
            use_reward_shaping=not args.no_reward_shaping,
            shaping_weight=phase.shaping_weight,
            eps_start=phase.eps_reset, eps_end=args.eps_end,
            eps_decay_episodes=phase_decay,
            gamma=args.gamma, batch_size=args.batch_size,
            target_update_steps=args.target_update_steps, grad_clip=args.grad_clip,
            log_interval=args.log_interval,
            eval_interval=curriculum.eval_interval,
            eval_episodes=curriculum.eval_episodes,
            save_interval=args.save_interval,
            save_dir=args.save_dir, render_eval=args.render_eval,
            phase_name=phase.name,
            on_eval=make_eval_callback(tracker),
        )

        total_episodes += result["episodes"]
        advanced_early = result["episodes"] < phase.max_episodes
        reason = "threshold met" if advanced_early else "max episodes reached"

        print(f"  Phase '{phase.name}' complete ({reason})")
        print(f"  Episodes: {result['episodes']}  │  "
              f"Phase best SR: {result['phase_best_sr']*100:.1f}%  │  "
              f"Global best SR: {result['global_best_sr']*100:.1f}%")
        print()

    # ── Final evaluation at target difficulty ──
    target_wall = curriculum.phases[-1].wall_length
    total_elapsed = time.time() - total_start

    print("=" * 70)
    print("  CURRICULUM TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total episodes:    {total_episodes}")
    print(f"  Total steps:       {ts.global_step:,}")
    print(f"  Total time:        {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Global best SR:    {ts.best_success_rate*100:.1f}%")

    print(f"\n  ── Final Evaluation (50 episodes, wall_length={target_wall}, greedy) ──")
    final = evaluate(q_net, encoder, device, n_episodes=50,
                     grid_size=args.size, max_steps=args.max_steps,
                     wall_length=target_wall)
    print(f"  Success={final['success_rate']*100:.1f}% │ "
          f"AvgSteps={final['avg_steps']:.1f} │ "
          f"Range=[{final['min_steps']}, {final['max_steps']}]")

    save_checkpoint(os.path.join(args.save_dir, "final_curriculum_model.pt"),
                    total_episodes, q_net, target_net, optimizer, args.eps_end,
                    ts.global_step, ts.best_success_rate,
                    extra={"curriculum_preset": args.preset})
    print(f"  [Final model → final_curriculum_model.pt]\nDone! 🎯")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DQN Training for GridWorld Navigation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                                    # Single-phase (defaults)
  python train.py --curriculum                       # Curriculum learning
  python train.py --curriculum --preset aggressive   # Faster curriculum
  python train.py --episodes 20000                   # More episodes (single)
  python train.py --no-reward-shaping                # Sparse reward only
  python train.py --resume checkpoints/best_model.pt # Resume training
  python train.py --render-eval                      # Visualize evals
        """,
    )

    # Mode
    mode = p.add_argument_group("Training Mode")
    mode.add_argument("--curriculum", action="store_true",
                      help="Enable curriculum learning (progressive difficulty)")
    mode.add_argument("--preset", type=str, default="default",
                      choices=["default", "aggressive"],
                      help="Curriculum preset (default: default)")

    # Environment
    env = p.add_argument_group("Environment")
    env.add_argument("--size", type=int, default=ENV_CONFIG["size"],
                     help=f"Grid size including borders (default: {ENV_CONFIG['size']})")
    env.add_argument("--max-steps", type=int, default=ENV_CONFIG["max_steps"],
                     help=f"Max steps per episode (default: {ENV_CONFIG['max_steps']})")
    env.add_argument("--wall-length", type=int, default=ENV_CONFIG["wall_length"],
                     help=f"Wall length for single-phase (default: {ENV_CONFIG['wall_length']})")

    # DQN hyperparameters
    dqn = p.add_argument_group("DQN Hyperparameters")
    dqn.add_argument("--lr", type=float, default=3e-4,
                     help="Learning rate (default: 3e-4)")
    dqn.add_argument("--gamma", type=float, default=AGENT_CONFIG["discount_factor"],
                     help=f"Discount factor (default: {AGENT_CONFIG['discount_factor']})")
    dqn.add_argument("--batch-size", type=int, default=256, help="Replay batch size (default: 256)")
    dqn.add_argument("--replay-size", type=int, default=100_000, help="Buffer capacity (default: 100000)")
    dqn.add_argument("--warmup-steps", type=int, default=5000,
                     help="Random steps before training (default: 5000)")
    dqn.add_argument("--target-update-steps", type=int, default=1000,
                     help="Steps between target net updates (default: 1000)")
    dqn.add_argument("--grad-clip", type=float, default=10.0, help="Max gradient norm (default: 10.0)")

    # Exploration
    exp = p.add_argument_group("Exploration")
    exp.add_argument("--eps-start", type=float, default=1.0, help="Initial epsilon (default: 1.0)")
    exp.add_argument("--eps-end", type=float, default=0.05, help="Final epsilon (default: 0.05)")
    exp.add_argument("--eps-decay-episodes", type=int, default=12_000,
                     help="Eps decay episodes for single-phase (default: 12000)")

    # Training
    trn = p.add_argument_group("Training")
    trn.add_argument("--episodes", type=int, default=TRAIN_CONFIG["n_episodes"],
                     help=f"Total episodes for single-phase (default: {TRAIN_CONFIG['n_episodes']})")
    trn.add_argument("--no-reward-shaping", action="store_true",
                     help="Disable reward shaping (sparse +1 only)")
    trn.add_argument("--shaping-weight", type=float, default=0.3,
                     help="Distance shaping weight for single-phase (default: 0.3)")
    trn.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")

    # Logging & saving
    log = p.add_argument_group("Logging & Saving")
    log.add_argument("--log-interval", type=int, default=50)
    log.add_argument("--eval-interval", type=int, default=500)
    log.add_argument("--eval-episodes", type=int, default=25)
    log.add_argument("--save-interval", type=int, default=TRAIN_CONFIG["save_interval"])
    log.add_argument("--save-dir", type=str, default=PATHS["save_dir"])
    log.add_argument("--render-eval", action="store_true")

    # Misc
    misc = p.add_argument_group("Misc")
    misc.add_argument("--seed", type=int, default=42)
    misc.add_argument("--cpu", action="store_true", help="Force CPU")

    return p.parse_args()


if __name__ == "__main__":
    args = build_args()
    if args.curriculum:
        train_curriculum(args)
    else:
        train_single(args)