"""Evaluation loop.

Evaluation should be isolated from training so you can:
  - run it from scripts
  - run it during training without clutter
  - adjust evaluation behavior (greedy, stochastic, rendered)
"""

from __future__ import annotations

import time

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as e:  # pragma: no cover
    raise ImportError("PyTorch is required. Install with: pip install torch") from e

from env import GridWorldEnv
from .encoder import StateEncoder


@torch.no_grad()
def evaluate(
    q_net: nn.Module,
    encoder: StateEncoder,
    device: torch.device,
    *,
    n_episodes: int = 25,
    grid_size: int = 12,
    max_steps: int = 120,
    wall_length: int = 4,
    render: bool = False,
) -> dict:
    """Greedy evaluation.

    Returns a dictionary so callers can log whatever they care about.
    """
    env = GridWorldEnv(
        size=grid_size,
        max_steps=max_steps,
        wall_length=wall_length,
        render_mode="human" if render else None,
    )

    q_net.eval()
    successes = 0
    steps_list: list[int] = []

    for _ in range(int(n_episodes)):
        obs, _ = env.reset()
        done = False
        steps = 0

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
        "success_rate": successes / max(1, int(n_episodes)),
        "avg_steps": float(np.mean(steps_list)) if steps_list else 0.0,
        "min_steps": int(min(steps_list)) if steps_list else 0,
        "max_steps": int(max(steps_list)) if steps_list else 0,
    }
