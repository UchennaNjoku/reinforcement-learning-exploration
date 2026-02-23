"""State encoding for GridWorld.

Your environment observation is a dictionary containing:
  - agent_pos: (2,) int32
  - goal_pos: (2,) int32
  - obstacle_positions: list[(2,) int32]

DQN expects a fixed-size vector, so we turn the observation into:
  - 3 spatial channels (walls, goal, agent) flattened
  - 7 scalar features (normalized positions and relative offsets)

This encoder is intentionally *pure*:
  - it does not touch PyTorch unless you ask for a tensor
  - it does not own any model parameters
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from env import GridWorldEnv


class StateEncoder:
    """Encodes a GridWorld observation into a flat float32 feature vector."""

    def __init__(self, grid_size: int):
        self.size = int(grid_size)
        # 3 channels (H*W each) + 7 scalar features
        self.feature_dim = 3 * self.size * self.size + 7

    def encode(self, obs: dict, env: GridWorldEnv) -> np.ndarray:
        """Return a 1D float32 feature vector of length `feature_dim`."""
        size = self.size
        assert env.grid is not None

        # --- Spatial channels -------------------------------------------------
        # Walls channel includes the border and internal obstacle tiles.
        ch_walls = (env.grid == GridWorldEnv.TILE_WALL).astype(np.float32)

        ch_goal = np.zeros((size, size), dtype=np.float32)
        ch_agent = np.zeros((size, size), dtype=np.float32)

        gx, gy = int(obs["goal_pos"][0]), int(obs["goal_pos"][1])
        ax, ay = int(obs["agent_pos"][0]), int(obs["agent_pos"][1])
        ch_goal[gy, gx] = 1.0
        ch_agent[ay, ax] = 1.0

        spatial = np.stack([ch_walls, ch_goal, ch_agent]).reshape(-1)

        # --- Scalar features --------------------------------------------------
        # Normalize by (size-1) so coordinates fit into [0,1].
        norm = size - 1
        # Manhattan distance normalized by max possible manhattan within playable
        max_manhattan = 2 * (size - 2)

        scalars = np.array(
            [
                ax / norm,
                ay / norm,
                gx / norm,
                gy / norm,
                (gx - ax) / norm,
                (gy - ay) / norm,
                (abs(gx - ax) + abs(gy - ay)) / max_manhattan,
            ],
            dtype=np.float32,
        )

        return np.concatenate([spatial, scalars])

    def encode_tensor(self, obs: dict, env: GridWorldEnv, device):
        """Encode and return a torch tensor shaped (1, D)."""
        if torch is None:
            raise ImportError("PyTorch is required for encode_tensor")
        x = self.encode(obs, env)
        return torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
