"""Replay-buffer warmup.

Warmup is the period before learning starts where we collect experience.
This prevents the first updates from training on an empty / tiny buffer.

Policies:
  - random: uniform random actions
  - bfs: mostly actions that reduce BFS distance to goal (with epsilon randomness)

BFS warmup is a nice hack for this env because it produces "reasonable" paths
without needing a trained policy.
"""

from __future__ import annotations

import random

from env import GridWorldEnv

from .distance import bfs_distance_map
from .replay import Transition, ReplayBuffer
from .encoder import StateEncoder


def warmup_buffer(
    replay: ReplayBuffer,
    encoder: StateEncoder,
    *,
    grid_size: int,
    max_steps: int,
    wall_length: int,
    n_steps: int,
    seed: int = 42,
    policy: str = "random",
    eps: float = 0.2,
) -> None:
    """Fill replay buffer with initial experience."""

    env = GridWorldEnv(
        size=grid_size,
        max_steps=max_steps,
        wall_length=wall_length,
        render_mode=None,
    )

    obs, _ = env.reset(seed=seed)

    dist_map = None
    if policy == "bfs":
        dist_map = bfs_distance_map(env, obs["goal_pos"], grid_size)

    for _ in range(int(n_steps)):
        # Choose action.
        if policy == "bfs" and dist_map is not None and random.random() > eps:
            ax, ay = map(int, obs["agent_pos"])
            best_a = None
            best_d = 10**9

            for a in range(env.action_space.n):
                nx, ny = ax, ay
                if a == 0:
                    ny -= 1
                elif a == 1:
                    ny += 1
                elif a == 2:
                    nx -= 1
                elif a == 3:
                    nx += 1

                # If blocked, the agent stays put.
                if env._is_blocked(nx, ny):
                    nx, ny = ax, ay

                d = int(dist_map[ny, nx])
                if d < best_d:
                    best_d = d
                    best_a = a

            action = int(best_a) if best_a is not None else int(env.action_space.sample())
        else:
            action = int(env.action_space.sample())

        next_obs, base_reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

        s = encoder.encode(obs, env)
        s2 = encoder.encode(next_obs, env)

        # Warmup uses base reward. (Shaping is for training.)
        replay.push(Transition(s=s, a=action, r=float(base_reward), s2=s2, done=done))

        obs = next_obs

        # When an episode ends during warmup, reset and (if bfs) recompute map.
        if done:
            obs, _ = env.reset()
            if policy == "bfs":
                dist_map = bfs_distance_map(env, obs["goal_pos"], grid_size)

    env.close()
