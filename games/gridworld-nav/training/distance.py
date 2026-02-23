"""Distance-map utilities (BFS).

We compute a *reverse* BFS distance map from the goal to every reachable tile.
This lets reward shaping use wall-aware distance instead of Euclidean distance.

Important: this runs once per episode (not every step).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from env import GridWorldEnv


def bfs_distance_map(env: GridWorldEnv, goal_xy, size: int) -> np.ndarray:
    """Reverse BFS from the goal.

    Args:
        env: GridWorldEnv (provides wall blocking logic)
        goal_xy: (x,y) goal coordinate
        size: grid size including border

    Returns:
        dist: int32 array shape (size,size)
              dist[y,x] = shortest steps from (x,y) to goal, respecting walls.
              Unreachable tiles are set to INF.
    """
    gx, gy = int(goal_xy[0]), int(goal_xy[1])

    INF = 10**9
    dist = np.full((size, size), INF, dtype=np.int32)

    q = deque([(gx, gy)])
    dist[gy, gx] = 0

    while q:
        x, y = q.popleft()
        d = int(dist[y, x])

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if env._is_blocked(nx, ny):
                continue
            if dist[ny, nx] > d + 1:
                dist[ny, nx] = d + 1
                q.append((nx, ny))

    return dist
