"""Replay buffer for the communicating agent.

Extends the baseline buffer with two extra fields:
  - received_msgs      : messages received from teammates at the current step
  - next_received_msgs : messages received after the transition

Both are stored as flat float32 one-hot vectors of size
(n_agents - 1) * vocab_size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class CommBatch:
    obs: torch.Tensor               # (B, 3, 7, 7)
    agent_ids: torch.Tensor         # (B,)
    received_msgs: torch.Tensor     # (B, (n_agents-1)*vocab_size)
    actions: torch.Tensor           # (B,)  joint action index
    rewards: torch.Tensor           # (B,)
    next_obs: torch.Tensor          # (B, 3, 7, 7)
    next_received_msgs: torch.Tensor  # (B, (n_agents-1)*vocab_size)
    dones: torch.Tensor             # (B,)


class CommReplayBuffer:
    """Fixed-capacity circular replay buffer for communicating agents.

    Args:
        capacity:    maximum stored transitions.
        obs_shape:   CHW obs shape, default (3, 7, 7).
        msg_dim:     (n_agents - 1) * vocab_size — total received-message dim.
        device:      torch device used when sampling.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        obs_shape: tuple[int, int, int] = (3, 7, 7),
        msg_dim: int = 8,            # (3-1) * 4 = 8 for vocab_size=4
        device: str | torch.device = "cpu",
    ) -> None:
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.msg_dim = msg_dim
        self.device = torch.device(device)
        self._ptr = 0
        self._size = 0

        self._obs           = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._next_obs      = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._agent_ids     = np.zeros(capacity, dtype=np.int64)
        self._recv_msgs     = np.zeros((capacity, msg_dim), dtype=np.float32)
        self._next_recv_msgs= np.zeros((capacity, msg_dim), dtype=np.float32)
        self._actions       = np.zeros(capacity, dtype=np.int64)
        self._rewards       = np.zeros(capacity, dtype=np.float32)
        self._dones         = np.zeros(capacity, dtype=np.float32)

    def push(
        self,
        obs: np.ndarray,
        agent_id: int,
        received_msgs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        next_received_msgs: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition. obs/next_obs are HWC; transposed to CHW."""
        self._obs[self._ptr]            = obs.transpose(2, 0, 1)
        self._next_obs[self._ptr]       = next_obs.transpose(2, 0, 1)
        self._agent_ids[self._ptr]      = agent_id
        self._recv_msgs[self._ptr]      = received_msgs
        self._next_recv_msgs[self._ptr] = next_received_msgs
        self._actions[self._ptr]        = action
        self._rewards[self._ptr]        = reward
        self._dones[self._ptr]          = float(done)

        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> CommBatch:
        if self._size < batch_size:
            raise ValueError(
                f"Buffer has {self._size} transitions, cannot sample {batch_size}."
            )
        idx = np.random.randint(0, self._size, size=batch_size)
        return CommBatch(
            obs=torch.from_numpy(self._obs[idx]).to(self.device),
            agent_ids=torch.from_numpy(self._agent_ids[idx]).to(self.device),
            received_msgs=torch.from_numpy(self._recv_msgs[idx]).to(self.device),
            actions=torch.from_numpy(self._actions[idx]).to(self.device),
            rewards=torch.from_numpy(self._rewards[idx]).to(self.device),
            next_obs=torch.from_numpy(self._next_obs[idx]).to(self.device),
            next_received_msgs=torch.from_numpy(self._next_recv_msgs[idx]).to(self.device),
            dones=torch.from_numpy(self._dones[idx]).to(self.device),
        )

    def __len__(self) -> int:
        return self._size
