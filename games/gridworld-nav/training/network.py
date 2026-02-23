"""Neural network definitions for DQN.

Keep networks in their own module so:
  - training loops stay readable
  - you can swap architectures without touching the algorithm code
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError as e:  # pragma: no cover
    raise ImportError("PyTorch is required. Install with: pip install torch") from e


class QNetwork(nn.Module):
    """Simple MLP Q-network.

    For GridWorld with a flattened spatial+scalar encoding, an MLP works fine.
    This is intentionally over-parameterized but stable.

    Input:  D
    Output: n_actions (default 4)
    """

    def __init__(self, input_dim: int, n_actions: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, int(n_actions)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
