"""
Utility functions for training and evaluation.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any


def setup_directories(paths: Dict[str, str]):
    """Create necessary directories if they don't exist."""
    for path in paths.values():
        if path.endswith(".pkl") or path.endswith(".png"):
            path = os.path.dirname(path)
        if path and not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")


def plot_training_stats(
    rewards: List[float],
    lengths: List[float],
    window: int = 100,
    save_path: str | None = None,
):
    """
    Plot training statistics with moving average.

    Args:
        rewards: List of episode rewards.
        lengths: List of episode lengths.
        window: Window size for moving average.
        save_path: Optional path to save the plot.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot rewards
    ax = axes[0]
    ax.plot(rewards, alpha=0.3, color="blue", label="Episode Reward")
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(
            range(window - 1, len(rewards)),
            moving_avg,
            color="red",
            label=f"Moving Avg ({window})",
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training Rewards")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot episode lengths
    ax = axes[1]
    ax.plot(lengths, alpha=0.3, color="green", label="Episode Length")
    if len(lengths) >= window:
        moving_avg = np.convolve(lengths, np.ones(window) / window, mode="valid")
        ax.plot(
            range(window - 1, len(lengths)),
            moving_avg,
            color="red",
            label=f"Moving Avg ({window})",
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title("Episode Lengths")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def print_episode_info(
    episode: int,
    reward: float,
    length: int,
    epsilon: float,
    q_table_size: int,
    recent_rewards: List[float],
    window: int = 100,
):
    """Print formatted episode information."""
    avg_reward = np.mean(recent_rewards[-window:]) if recent_rewards else 0
    print(
        f"Episode {episode:5d} | "
        f"Reward: {reward:7.2f} | "
        f"Avg Reward: {avg_reward:7.2f} | "
        f"Steps: {length:3d} | "
        f"Epsilon: {epsilon:.4f} | "
        f"Q-States: {q_table_size}"
    )


def discretize_state(
    agent_pos: tuple,
    goal_pos: tuple,
    grid_size: int = 12,
) -> tuple:
    """
    Create a discrete state representation for tabular Q-learning.

    Args:
        agent_pos: (x, y) position of agent.
        goal_pos: (x, y) position of goal.
        grid_size: Size of the grid.

    Returns:
        A tuple representing the state.
    """
    # Simple state: just positions
    # Can be extended to include relative direction, distance, etc.
    return (int(agent_pos[0]), int(agent_pos[1]), int(goal_pos[0]), int(goal_pos[1]))


def get_relative_direction(
    agent_pos: tuple,
    agent_dir: int,  # 0=right, 1=down, 2=left, 3=up
    target_pos: tuple,
) -> int:
    """
    Get relative direction to target from agent's perspective.

    Returns:
        0: target is ahead
        1: target is to the right
        2: target is behind
        3: target is to the left
    """
    dx = target_pos[0] - agent_pos[0]
    dy = target_pos[1] - agent_pos[1]

    # Determine absolute direction to target
    if abs(dx) > abs(dy):
        # Target is more horizontal
        target_dir = 0 if dx > 0 else 2  # right or left
    else:
        # Target is more vertical
        target_dir = 1 if dy > 0 else 3  # down or up

    # Calculate relative direction
    rel_dir = (target_dir - agent_dir) % 4
    return rel_dir
