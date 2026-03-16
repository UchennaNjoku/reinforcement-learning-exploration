"""Communicating Q-network for discrete emergent communication.

Architecture:
  - Same CNN backbone as PursuitQNet (obs → 3136 features)
  - Agent-ID embedding (same as baseline)
  - Received-message encoder: (n_agents-1) one-hot vectors → embedding
  - Head → Q-values over joint (move, message) action space

Joint action space:
  - Total actions = n_move_actions × vocab_size
  - action_idx = move_idx * vocab_size + msg_idx
  - Decode: move = action_idx // vocab_size
             msg  = action_idx  % vocab_size

Message timing (per specs):
  - At step t each agent receives the messages its teammates sent at t-1.
  - At episode start, all received messages are zeros.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CommQNet(nn.Module):
    """Shared-parameter communicating Q-network.

    Args:
        n_agents:        number of pursuers (default 3).
        n_move_actions:  number of movement actions (default 5).
        vocab_size:      discrete message vocabulary size (4 or 16).
        agent_emb_dim:   agent-ID embedding dimension.
        msg_emb_dim:     message encoder output dimension.
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
        self.n_joint_actions = n_move_actions * vocab_size

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

        # Message encoder: received (n_agents-1) one-hot vectors concatenated
        msg_input_dim = (n_agents - 1) * vocab_size
        self.msg_encoder = nn.Sequential(
            nn.Linear(msg_input_dim, msg_emb_dim),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(cnn_out_dim + agent_emb_dim + msg_emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_joint_actions),
        )

    def forward(
        self,
        obs: torch.Tensor,
        agent_ids: torch.Tensor,
        received_msgs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Q-values over joint (move, message) actions.

        Args:
            obs:           (B, 3, 7, 7) float32
            agent_ids:     (B,) long — values in {0, …, n_agents-1}
            received_msgs: (B, (n_agents-1) * vocab_size) float32 — one-hot

        Returns:
            Q-values of shape (B, n_move_actions * vocab_size)
        """
        cnn_feat = self.cnn(obs)                         # (B, 3136)
        agent_emb = self.agent_embedding(agent_ids)      # (B, agent_emb_dim)
        msg_emb = self.msg_encoder(received_msgs)        # (B, msg_emb_dim)
        combined = torch.cat([cnn_feat, agent_emb, msg_emb], dim=1)
        return self.head(combined)                       # (B, n_joint_actions)

    def decode_action(self, joint_action: int) -> tuple[int, int]:
        """Split a joint action index into (move, message) indices."""
        return divmod(joint_action, self.vocab_size)
