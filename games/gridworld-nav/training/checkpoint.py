"""Checkpoint save/load.

We save enough state to resume training:
  - online Q-network weights
  - target network weights
  - optimizer state
  - episode counter / global step
  - epsilon value (useful for logging)
  - best success rate so far

Note on torch.load(weights_only=...)
-----------------------------------
`weights_only` is a newer PyTorch argument and may not exist in older versions.
We keep compatibility by trying it and falling back when needed.
"""

from __future__ import annotations

import os
from typing import Any

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError("PyTorch is required. Install with: pip install torch") from e


def save_checkpoint(
    filepath: str,
    *,
    episode: int,
    q_net,
    target_net,
    optimizer,
    epsilon: float,
    global_step: int,
    best_success_rate: float,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a training checkpoint."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    data: dict[str, Any] = {
        "episode": int(episode),
        "q_state_dict": q_net.state_dict(),
        "target_state_dict": target_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epsilon": float(epsilon),
        "global_step": int(global_step),
        "best_success_rate": float(best_success_rate),
    }
    if extra:
        data.update(extra)

    torch.save(data, filepath)


def load_checkpoint(filepath: str, q_net, target_net, optimizer, device):
    """Load checkpoint and restore weights/optimizer.

    Returns:
        ckpt dict (caller can read episode/global_step/etc.)
    """
    # Try newer PyTorch signature first.
    try:
        ckpt = torch.load(filepath, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(filepath, map_location=device)

    q_net.load_state_dict(ckpt["q_state_dict"])
    target_net.load_state_dict(ckpt["target_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt
