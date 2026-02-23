"""
Custom GridWorld Environment: Cardinal Direction Navigation
===========================================================
A Gymnasium environment where the agent moves in 4 cardinal directions.
No orientation model - agent simply moves up/down/left/right.

Features:
- 10x10 playable grid (12x12 with border walls)
- Random agent start position
- Random goal position
- Blocking wall obstacle placed between agent and goal
- Reachability guarantee via BFS check

Actions:
    0 = UP    (decrease y) (row index)
    1 = DOWN  (increase y)
    2 = LEFT  (decrease x)
    3 = RIGHT (increase x)

"""

from environment.gridworld_env import GridWorldEnv  # noqa: F401
