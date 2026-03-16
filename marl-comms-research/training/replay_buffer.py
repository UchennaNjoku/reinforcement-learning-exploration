"""Circular replay buffer for the shared-parameter DQN baseline.

Each transition stores one (agent, obs, action, reward, next_obs, done) tuple.
Because all agents share one network, transitions from all pursuers are mixed
into the same buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Batch:
    obs: torch.Tensor        # (B, 3, 7, 7)  float32
    agent_ids: torch.Tensor  # (B,)           long
    actions: torch.Tensor    # (B,)           long
    rewards: torch.Tensor    # (B,)           float32
    next_obs: torch.Tensor   # (B, 3, 7, 7)  float32
    dones: torch.Tensor      # (B,)           float32


class ReplayBuffer:
    """Fixed-capacity circular replay buffer.

    Args:
        capacity: maximum number of stored transitions.
        obs_shape: observation shape as stored (C, H, W) = (3, 7, 7).
        device: torch device used when sampling.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        obs_shape: tuple[int, int, int] = (3, 7, 7),
        device: str | torch.device = "cpu",
    ) -> None:
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.device = torch.device(device)
        self._ptr = 0
        self._size = 0

        self._obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._agent_ids = np.zeros(capacity, dtype=np.int64)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)

    def push(
        self,
        obs: np.ndarray,
        agent_id: int,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition.

        obs / next_obs should be (7, 7, 3) HWC arrays as returned by the env;
        they are transposed to CHW on insertion.
        """
        self._obs[self._ptr] = obs.transpose(2, 0, 1)
        self._next_obs[self._ptr] = next_obs.transpose(2, 0, 1)
        self._agent_ids[self._ptr] = agent_id
        self._actions[self._ptr] = action
        self._rewards[self._ptr] = reward
        self._dones[self._ptr] = float(done)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        """Sample a random mini-batch."""
        if self._size < batch_size:
            raise ValueError(
                f"Buffer has {self._size} transitions, cannot sample {batch_size}."
            )
        idx = np.random.randint(0, self._size, size=batch_size)
        return Batch(
            obs=torch.from_numpy(self._obs[idx]).to(self.device),
            agent_ids=torch.from_numpy(self._agent_ids[idx]).to(self.device),
            actions=torch.from_numpy(self._actions[idx]).to(self.device),
            rewards=torch.from_numpy(self._rewards[idx]).to(self.device),
            next_obs=torch.from_numpy(self._next_obs[idx]).to(self.device),
            dones=torch.from_numpy(self._dones[idx]).to(self.device),
        )

    def __len__(self) -> int:
        return self._size
