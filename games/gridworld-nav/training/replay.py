"""Replay buffer utilities.

Why this file exists
--------------------
The replay buffer is an algorithm component (DQN) and should *not* live inside
"train.py". Keeping it here makes the training loop easier to read and lets
other scripts (eval, ablations, unit tests) reuse the same buffer.

We keep the stored transition as NumPy arrays / Python primitives so that:
  - replay is device-agnostic (CPU/GPU doesn't matter)
  - sampling is fast
  - torch tensors are created only at the update step
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Transition:
    """A single experience tuple."""

    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-size circular replay buffer."""

    def __init__(self, capacity: int):
        self._buf: deque[Transition] = deque(maxlen=int(capacity))

    def push(self, t: Transition) -> None:
        self._buf.append(t)

    def sample(self, batch_size: int):
        """Uniform random sample.

        Returns:
            s:   (B, D) float32
            a:   (B,)   int64
            r:   (B,)   float32
            s2:  (B, D) float32
            done:(B,)   float32 (1.0 if done else 0.0)
        """
        batch = random.sample(self._buf, int(batch_size))
        s = np.stack([b.s for b in batch]).astype(np.float32, copy=False)
        a = np.array([b.a for b in batch], dtype=np.int64)
        r = np.array([b.r for b in batch], dtype=np.float32)
        s2 = np.stack([b.s2 for b in batch]).astype(np.float32, copy=False)
        done = np.array([b.done for b in batch], dtype=np.float32)
        return s, a, r, s2, done

    def __len__(self) -> int:
        return len(self._buf)
