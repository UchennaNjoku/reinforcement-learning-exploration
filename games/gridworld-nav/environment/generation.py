"""Procedural generation utilities for GridWorld.

Keeps the Gym env file small by isolating:
  - wall placement
  - reachability (BFS)
  - fallback map
"""

from __future__ import annotations

from collections import deque
from typing import List, Tuple, Optional

import numpy as np

from .constants import TILE_EMPTY, TILE_WALL, TILE_GOAL


def is_valid_pos(size: int, x: int, y: int) -> bool:
    """Playable area excludes the outer border walls."""
    return 1 <= x < size - 1 and 1 <= y < size - 1


def is_blocked(grid: np.ndarray, size: int, x: int, y: int) -> bool:
    if not is_valid_pos(size, x, y):
        return True
    return grid[y, x] == TILE_WALL


def bfs_reachable(grid: np.ndarray, size: int, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
    """BFS: can we get from start to goal using 4-neighborhood moves?"""
    q = deque([start])
    visited = {start}

    while q:
        x, y = q.popleft()
        if (x, y) == goal:
            return True

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if not is_valid_pos(size, nx, ny):
                continue
            if (nx, ny) in visited:
                continue
            if is_blocked(grid, size, nx, ny):
                continue
            visited.add((nx, ny))
            q.append((nx, ny))

    return False


def place_blocking_wall(
    grid: np.ndarray,
    size: int,
    agent_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
    wall_length: int,
    wall_position_range: Tuple[float, float],
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    """Place a wall roughly perpendicular to the agent->goal vector.

    Returns a list of placed tiles of EXACT length = wall_length, or [] if it fails.
    """
    t_min, t_max = wall_position_range
    t = float(rng.uniform(t_min, t_max))

    ax, ay = agent_pos
    gx, gy = goal_pos

    wall_x = int(round(ax + t * (gx - ax)))
    wall_y = int(round(ay + t * (gy - ay)))

    delta_x = abs(gx - ax)
    delta_y = abs(gy - ay)
    path_is_more_horizontal = delta_x > delta_y

    L = int(wall_length)
    start = -(L // 2)
    offsets = list(range(start, start + L))  # length L always

    attempts = [
        (wall_x, wall_y),
        (wall_x + 1, wall_y),
        (wall_x - 1, wall_y),
        (wall_x, wall_y + 1),
        (wall_x, wall_y - 1),
    ]

    for cx, cy in attempts:
        placed: List[Tuple[int, int]] = []
        for off in offsets:
            if path_is_more_horizontal:
                wx, wy = cx, cy + off  # vertical wall
            else:
                wx, wy = cx + off, cy  # horizontal wall

            if not is_valid_pos(size, wx, wy):
                continue
            if (wx, wy) == agent_pos or (wx, wy) == goal_pos:
                continue
            if grid[wy, wx] != TILE_EMPTY:
                continue

            grid[wy, wx] = TILE_WALL
            placed.append((wx, wy))

            if len(placed) == L:
                return placed

        # undo partial placement
        for wx, wy in placed:
            grid[wy, wx] = TILE_EMPTY

    return []


def create_fallback_grid(size: int) -> tuple[np.ndarray, Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]:
    """Guaranteed-solvable simple configuration."""
    grid = np.full((size, size), TILE_EMPTY, dtype=np.int32)
    grid[0, :] = TILE_WALL
    grid[-1, :] = TILE_WALL
    grid[:, 0] = TILE_WALL
    grid[:, -1] = TILE_WALL

    agent_pos = (1, size - 2)
    goal_pos = (size - 2, 1)
    grid[goal_pos[1], goal_pos[0]] = TILE_GOAL
    obstacles: List[Tuple[int, int]] = []
    return grid, agent_pos, goal_pos, obstacles


def generate_grid(
    size: int,
    wall_length: int,
    wall_position_range: Tuple[float, float],
    rng: np.random.Generator,
    max_retries: int = 100,
) -> tuple[np.ndarray, Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]:
    """Generate a random grid with a blocking wall and reachability guarantee."""
    for _ in range(max_retries):
        grid = np.full((size, size), TILE_EMPTY, dtype=np.int32)

        # border walls
        grid[0, :] = TILE_WALL
        grid[-1, :] = TILE_WALL
        grid[:, 0] = TILE_WALL
        grid[:, -1] = TILE_WALL

        # goal
        gx = int(rng.integers(1, size - 1))
        gy = int(rng.integers(1, size - 1))
        goal_pos = (gx, gy)
        grid[gy, gx] = TILE_GOAL

        # agent (not on goal)
        while True:
            ax = int(rng.integers(1, size - 1))
            ay = int(rng.integers(1, size - 1))
            if (ax, ay) != goal_pos:
                agent_pos = (ax, ay)
                break

        obstacles = place_blocking_wall(
            grid=grid,
            size=size,
            agent_pos=agent_pos,
            goal_pos=goal_pos,
            wall_length=wall_length,
            wall_position_range=wall_position_range,
            rng=rng,
        )

        # sanity: goal isn't on a wall tile
        if goal_pos in obstacles:
            continue

        # reachability
        if not bfs_reachable(grid, size, agent_pos, goal_pos):
            continue

        return grid, agent_pos, goal_pos, obstacles

    # last resort fallback
    return create_fallback_grid(size)
