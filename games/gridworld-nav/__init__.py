"""
GridWorld Navigation - Game 3
==============================
A Gymnasium environment for cardinal direction grid navigation.

This package provides:
- GridWorldEnv: Custom environment with 4-directional movement
- Manual play interface for testing
- Configuration and utilities

Usage:
    from envs import GridWorldEnv
    
    env = GridWorldEnv(size=12, render_mode="human")
    obs, info = env.reset()
    
    for _ in range(100):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    env.close()
"""

__version__ = "1.0.0"

# Make envs accessible
from envs import GridWorldEnv

__all__ = ["GridWorldEnv"]
