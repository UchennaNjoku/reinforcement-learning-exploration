"""Shared-parameter Q-network for the no-communication baseline.

Architecture:
  - CNN over (7, 7, 3) local observation
  - Agent-ID embedding concatenated with CNN features
  - Two FC layers → Q-values for 5 move actions

Parameter sharing: all pursuers use one instance of this network.
The agent index (0, 1, 2) is passed as input so the network can
differentiate between agents without separate weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PursuitQNet(nn.Module):
    """CNN Q-network with agent-ID embedding.

    Args:
        n_agents: number of pursuers (default 3).
        n_actions: move action count (default 5).
        agent_emb_dim: dimensionality of the agent-ID embedding.
    """

    def __init__(
        self,
        n_agents: int = 3,
        n_actions: int = 5,
        agent_emb_dim: int = 8,
    ) -> None:
        super().__init__()

        # CNN processes the (C, H, W) = (3, 7, 7) observation.
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),  # → 64 * 7 * 7 = 3136
        )

        cnn_out_dim = 64 * 7 * 7  # 3136

        self.agent_embedding = nn.Embedding(n_agents, agent_emb_dim)

        self.head = nn.Sequential(
            nn.Linear(cnn_out_dim + agent_emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, obs: torch.Tensor, agent_ids: torch.Tensor) -> torch.Tensor:
        """Compute Q-values.

        Args:
            obs: float tensor of shape (B, 3, 7, 7), values in [0, 3].
            agent_ids: long tensor of shape (B,), values in {0, 1, 2}.

        Returns:
            Q-values of shape (B, n_actions).
        """
        cnn_features = self.cnn(obs)                      # (B, 3136)
        agent_emb = self.agent_embedding(agent_ids)       # (B, agent_emb_dim)
        combined = torch.cat([cnn_features, agent_emb], dim=1)
        return self.head(combined)                        # (B, n_actions)
