"""Core DQN training loop (single phase).

This module contains *only* the algorithm loop for one fixed difficulty
(e.g., a specific wall_length). Curriculum logic lives elsewhere.

Design goals
------------
- Keep the control flow readable.
- Separate concerns:
    * env interaction
    * replay storage
    * network update
    * evaluation / checkpoint hooks
"""

from __future__ import annotations

import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as e:  # pragma: no cover
    raise ImportError("PyTorch is required. Install with: pip install torch") from e

from env import GridWorldEnv

from .checkpoint import save_checkpoint
from .distance import bfs_distance_map
from .encoder import StateEncoder
from .eval import evaluate
from .replay import ReplayBuffer, Transition
from .reward import shaped_reward
from .schedules import linear_epsilon


@dataclass
class TrainState:
    """Mutable state carried across phases (curriculum) or runs."""

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
    # Exploration
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
    # Optional metadata
    phase_name: str = "",
    # Curriculum hook: called after eval; return True to stop phase early.
    on_eval: Optional[Callable[[float], bool]] = None,
) -> dict:
    """Train for `n_episodes` at a fixed wall_length.

    Returns summary dict:
      episodes, phase_best_sr, global_best_sr, global_step, elapsed
    """

    env = GridWorldEnv(size=grid_size, max_steps=max_steps, wall_length=wall_length, render_mode=None)

    q_net = state.q_net
    target_net = state.target_net
    optimizer = state.optimizer
    replay = state.replay
    encoder = state.encoder
    device = state.device

    os.makedirs(save_dir, exist_ok=True)

    # Rolling stats for logs.
    recent_rewards = deque(maxlen=100)
    recent_successes = deque(maxlen=100)
    recent_losses = deque(maxlen=500)

    start_time = time.time()

    # Only compute BFS maps if we're actually going to use them.
    need_dist_map = bool(use_reward_shaping and shaping_weight > 0.0)

    phase_best_sr = 0.0
    total_ep = 0

    for ep in range(1, int(n_episodes) + 1):
        obs, _ = env.reset()
        prev_obs = obs
        s = encoder.encode(obs, env)

        # Compute BFS distance map ONCE per episode.
        dist_map = bfs_distance_map(env, obs["goal_pos"], grid_size) if need_dist_map else None

        ep_reward = 0.0
        steps = 0
        done = False
        reached_goal = False

        eps = linear_epsilon(ep, eps_start, eps_end, eps_decay_episodes)

        while not done:
            state.global_step += 1
            steps += 1

            # --- Action selection (epsilon-greedy) ---------------------------
            if random.random() < eps:
                action = int(env.action_space.sample())
            else:
                with torch.no_grad():
                    st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                    action = int(q_net(st).argmax(dim=1).item())

            # --- Env step -----------------------------------------------------
            next_obs, base_reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            if terminated:
                reached_goal = True

            # --- Reward (sparse vs shaped) -----------------------------------
            if use_reward_shaping:
                reward = shaped_reward(
                    next_obs,
                    prev_obs,
                    terminated,
                    truncated,
                    dist_map,
                    grid_size,
                    shaping_weight=shaping_weight,
                )
            else:
                reward = float(base_reward)

            # --- Store transition --------------------------------------------
            s2 = encoder.encode(next_obs, env)
            replay.push(Transition(s=s, a=action, r=reward, s2=s2, done=done))

            # --- Learning update (Double DQN) --------------------------------
            if len(replay) >= batch_size:
                bs, ba, br, bs2, bdone = replay.sample(batch_size)

                bs_t = torch.tensor(bs, dtype=torch.float32, device=device)
                ba_t = torch.tensor(ba, dtype=torch.long, device=device).unsqueeze(1)
                br_t = torch.tensor(br, dtype=torch.float32, device=device)
                bs2_t = torch.tensor(bs2, dtype=torch.float32, device=device)
                bdone_t = torch.tensor(bdone, dtype=torch.float32, device=device)

                # Q(s,a) for the actions actually taken.
                q_vals = q_net(bs_t).gather(1, ba_t).squeeze(1)

                # Double DQN target:
                #   - online net chooses next action
                #   - target net evaluates that chosen action
                with torch.no_grad():
                    next_a = q_net(bs2_t).argmax(dim=1, keepdim=True)
                    next_q = target_net(bs2_t).gather(1, next_a).squeeze(1)
                    target_q = br_t + (1.0 - bdone_t) * gamma * next_q

                loss = nn.SmoothL1Loss()(q_vals, target_q)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), grad_clip)
                optimizer.step()

                recent_losses.append(float(loss.item()))

                # Hard update target network every N steps.
                if state.global_step % int(target_update_steps) == 0:
                    target_net.load_state_dict(q_net.state_dict())

            # Move forward.
            s = s2
            prev_obs = next_obs
            ep_reward += reward

        # --- End of episode bookkeeping --------------------------------------
        total_ep = ep
        recent_rewards.append(ep_reward)
        recent_successes.append(1.0 if reached_goal else 0.0)

        # --- Logging ----------------------------------------------------------
        if log_interval > 0 and ep % int(log_interval) == 0:
            avg_r = float(np.mean(recent_rewards)) if recent_rewards else 0.0
            avg_succ = float(np.mean(recent_successes) * 100.0) if recent_successes else 0.0
            avg_loss = float(np.mean(recent_losses)) if recent_losses else 0.0

            tag = f"[{phase_name}] " if phase_name else ""
            print(
                f"  {tag}Ep {ep:>5d}/{n_episodes} │ "
                f"R={avg_r:>7.2f} │ "
                f"Succ={avg_succ:>5.1f}% │ "
                f"ε={eps:.4f} │ "
                f"Loss={avg_loss:.4f} │ "
                f"buf={len(replay):>6d}"
            )

        # --- Evaluation + (optional) early stop ------------------------------
        should_stop = False
        if eval_interval > 0 and ep % int(eval_interval) == 0:
            ev = evaluate(
                q_net,
                encoder,
                device,
                n_episodes=eval_episodes,
                grid_size=grid_size,
                max_steps=max_steps,
                wall_length=wall_length,
                render=render_eval,
            )

            sr = float(ev["success_rate"])
            phase_best_sr = max(phase_best_sr, sr)

            tag = f"[{phase_name}] " if phase_name else ""
            print(
                f"\n  {tag}── EVAL @Ep {ep} ──  "
                f"Success={sr*100:.1f}% │ "
                f"AvgSteps={ev['avg_steps']:.1f} │ "
                f"Range=[{ev['min_steps']}, {ev['max_steps']}]\n"
            )

            # Save a "global best" model, but avoid saving during easy phases.
            if sr > state.best_success_rate and wall_length >= 3:
                state.best_success_rate = sr
                save_checkpoint(
                    os.path.join(save_dir, "best_model.pt"),
                    episode=ep,
                    q_net=q_net,
                    target_net=target_net,
                    optimizer=optimizer,
                    epsilon=eps,
                    global_step=state.global_step,
                    best_success_rate=state.best_success_rate,
                    extra={"phase": phase_name, "wall_length": wall_length},
                )
                print(f"  ★ New global best: {sr*100:.1f}% → saved best_model.pt\n")

            # Curriculum callback: if it returns True, stop the phase early.
            if on_eval is not None:
                should_stop = bool(on_eval(sr))
                if should_stop:
                    print("  ✓ Phase advancement threshold met!\n")

        # --- Periodic checkpoint ---------------------------------------------
        if save_interval > 0 and ep % int(save_interval) == 0:
            save_checkpoint(
                os.path.join(save_dir, f"checkpoint_step{state.global_step}.pt"),
                episode=ep,
                q_net=q_net,
                target_net=target_net,
                optimizer=optimizer,
                epsilon=eps,
                global_step=state.global_step,
                best_success_rate=state.best_success_rate,
                extra={"phase": phase_name, "wall_length": wall_length},
            )

        if should_stop:
            break

    env.close()
    elapsed = float(time.time() - start_time)

    return {
        "episodes": int(total_ep),
        "phase_best_sr": float(phase_best_sr),
        "global_best_sr": float(state.best_success_rate),
        "global_step": int(state.global_step),
        "elapsed": elapsed,
    }
