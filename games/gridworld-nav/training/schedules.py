"""Schedules (epsilon, etc.)."""

from __future__ import annotations


def linear_epsilon(episode: int, start: float, end: float, decay_episodes: int) -> float:
    """Linear epsilon decay.

    - At episode 0: epsilon = start
    - At episode decay_episodes: epsilon = end
    - After that: epsilon stays at end
    """
    episode = int(episode)
    decay_episodes = max(1, int(decay_episodes))
    if episode >= decay_episodes:
        return float(end)
    t = episode / decay_episodes
    return float(start + (end - start) * t)
