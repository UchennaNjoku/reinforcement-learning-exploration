"""Reward shaping.

GridWorld is sparse reward by default (+1 only when reaching the goal).
That can be slow early on, so we optionally add *potential-based shaping*
using the BFS distance map.

Potential-based shaping idea (informal):
  - you get a small positive reward when you reduce distance-to-goal
  - you get a small negative reward when you increase distance-to-goal

We normalize by max_dist so the shaping reward scale stays consistent.
"""

from __future__ import annotations

import numpy as np


def shaped_reward(
    obs: dict,
    prev_obs: dict,
    terminated: bool,
    truncated: bool,
    dist_map: np.ndarray | None,
    grid_size: int,
    shaping_weight: float = 1.0,
) -> float:
    """Compute shaped reward.

    Reward components:
        +5.0    goal reached
        -0.01   step penalty
        ±w*Δd   BFS-distance improvement (normalized)
        -0.03   wall-bump penalty (agent didn't move)
        -0.5    truncation penalty (ran out of steps)

    Args:
        obs: current env observation dict
        prev_obs: previous env observation dict
        terminated: env terminated flag
        truncated: env truncated flag
        dist_map: BFS distance map (or None)
        grid_size: env size (including border)
        shaping_weight: 0 disables shaping

    Returns:
        float reward
    """
    if terminated:
        return 5.0

    reward = -0.01

    if shaping_weight > 0.0 and dist_map is not None:
        pax, pay = int(prev_obs["agent_pos"][0]), int(prev_obs["agent_pos"][1])
        cax, cay = int(obs["agent_pos"][0]), int(obs["agent_pos"][1])

        prev_d = int(dist_map[pay, pax])
        curr_d = int(dist_map[cay, cax])

        # Maximum manhattan distance in playable area is 2*(size-2)
        max_dist = 2 * (grid_size - 2)
        reward += shaping_weight * (prev_d - curr_d) / max_dist

    # If agent didn't move, it likely hit a wall.
    if (
        obs["agent_pos"][0] == prev_obs["agent_pos"][0]
        and obs["agent_pos"][1] == prev_obs["agent_pos"][1]
    ):
        reward -= 0.03

    if truncated:
        reward -= 0.5

    return float(reward)
