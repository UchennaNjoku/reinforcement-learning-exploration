"""Communicating Q-network with separate move and message heads.

Architecture:
  - CNN backbone (identical to PursuitQNet)
  - Agent-ID embedding
  - Received-message encoder: (n_agents-1) one-hot vectors → embedding
  - Shared trunk: combined features → 256 → 128
  - Move head:    128 → n_move_actions   (trained with DQN)
  - Message head: 128 → vocab_size       (trained with same TD target)

Why separate heads (not joint Q-values):
  Joint Q(move, message) over vocab_size*5 actions fails in practice because
  DQN cannot cleanly separate move credit from message credit across a large
  action space with sparse rewards. Separate heads keep move selection
  identical in complexity to the baseline (5 Q-values), while the message
  head learns which symbols are associated with higher future team reward.

Message timing (per specs):
  At step t each agent receives its teammates' messages from step t-1.
  At episode start all received messages are zero vectors.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CommQNet(nn.Module):
    """Shared-parameter communicating Q-network with dual output heads.

    Args:
        n_agents:       number of pursuers (default 3).
        n_move_actions: movement action count (default 5).
        vocab_size:     discrete message vocabulary size (4 or 16).
        agent_emb_dim:  agent-ID embedding dimension.
        msg_emb_dim:    received-message encoder output dimension.
    """

    def __init__(
        self,
        n_agents: int = 3,
        n_move_actions: int = 5,
        vocab_size: int = 4,
        agent_emb_dim: int = 8,
        msg_emb_dim: int = 16,
    ) -> None:
        super().__init__()

        self.n_agents = n_agents
        self.n_move_actions = n_move_actions
        self.vocab_size = vocab_size

        # CNN — identical to baseline
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),  # → 64 * 7 * 7 = 3136
        )
        cnn_out_dim = 64 * 7 * 7  # 3136

        self.agent_embedding = nn.Embedding(n_agents, agent_emb_dim)

        # Message encoder: (n_agents-1) one-hot vectors concatenated
        msg_input_dim = (n_agents - 1) * vocab_size
        self.msg_encoder = nn.Sequential(
            nn.Linear(msg_input_dim, msg_emb_dim),
            nn.ReLU(),
        )

        # Shared trunk
        trunk_in = cnn_out_dim + agent_emb_dim + msg_emb_dim
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Independent output heads
        self.move_head = nn.Linear(128, n_move_actions)
        self.msg_head  = nn.Linear(128, vocab_size)

    def forward(
        self,
        obs: torch.Tensor,
        agent_ids: torch.Tensor,
        received_msgs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute move and message Q-values.

        Args:
            obs:           (B, 3, 7, 7) float32
            agent_ids:     (B,) long
            received_msgs: (B, (n_agents-1) * vocab_size) float32 — one-hot

        Returns:
            move_q: (B, n_move_actions)
            msg_q:  (B, vocab_size)
        """
        cnn_feat  = self.cnn(obs)
        agent_emb = self.agent_embedding(agent_ids)
        msg_emb   = self.msg_encoder(received_msgs)
        combined  = torch.cat([cnn_feat, agent_emb, msg_emb], dim=1)
        trunk_out = self.trunk(combined)
        return self.move_head(trunk_out), self.msg_head(trunk_out)
