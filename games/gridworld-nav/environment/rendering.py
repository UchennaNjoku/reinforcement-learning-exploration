"""Rendering helpers (pygame + ANSI) for GridWorld.

This file is the ONLY place that knows about pygame.
The Gym env should not import pygame directly.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .constants import TILE_WALL, TILE_GOAL


def render_text(grid: np.ndarray, agent_pos: tuple[int, int], goal_pos: tuple[int, int]) -> str:
    size = grid.shape[0]
    lines = []
    for y in range(size):
        row = ""
        for x in range(size):
            if (x, y) == agent_pos:
                row += " A"
            elif (x, y) == goal_pos:
                row += " G"
            elif grid[y, x] == TILE_WALL:
                row += " #"
            else:
                row += " ."
        lines.append(row)
    return "\n".join(lines)


def render_frame(
    grid: np.ndarray,
    agent_pos: tuple[int, int],
    goal_pos: tuple[int, int],
    *,
    cell_size: int,
    render_mode: str,
    fps: int,
    window: Optional[object],
    clock: Optional[object],
) -> tuple[np.ndarray | None, Optional[object], Optional[object]]:
    """Render using pygame.

    Returns (rgb_array_or_none, window, clock).
    """
    try:
        import pygame  # type: ignore
    except ImportError:
        print("pygame not installed. Install with: pip install pygame")
        h = grid.shape[0] * cell_size
        w = grid.shape[1] * cell_size
        return np.zeros((h, w, 3), dtype=np.uint8), window, clock

    size = grid.shape[0]

    if window is None and render_mode == "human":
        pygame.init()
        pygame.display.init()
        window = pygame.display.set_mode((size * cell_size, size * cell_size))
        pygame.display.set_caption("GridWorld Navigation")

    if clock is None and render_mode == "human":
        clock = pygame.time.Clock()

    canvas = pygame.Surface((size * cell_size, size * cell_size))
    canvas.fill((200, 200, 200))

    # Colors (simple, readable)
    COLOR_WALL = (64, 64, 64)
    COLOR_GOAL = (0, 200, 0)
    COLOR_AGENT = (0, 0, 200)
    COLOR_EMPTY = (240, 240, 240)
    COLOR_GRID = (180, 180, 180)

    for y in range(size):
        for x in range(size):
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            tile = int(grid[y, x])
            if tile == TILE_WALL:
                pygame.draw.rect(canvas, COLOR_WALL, rect)
            elif tile == TILE_GOAL:
                pygame.draw.rect(canvas, COLOR_GOAL, rect)
            else:
                pygame.draw.rect(canvas, COLOR_EMPTY, rect)
            pygame.draw.rect(canvas, COLOR_GRID, rect, 1)

    ax, ay = agent_pos
    center = (ax * cell_size + cell_size // 2, ay * cell_size + cell_size // 2)
    radius = cell_size // 3
    pygame.draw.circle(canvas, COLOR_AGENT, center, radius)

    if render_mode == "human":
        window.blit(canvas, (0, 0))
        pygame.event.pump()
        pygame.display.flip()
        clock.tick(fps)
        return None, window, clock

    # rgb_array
    rgb = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
    return rgb, window, clock


def close_window(window: Optional[object], clock: Optional[object]) -> tuple[Optional[object], Optional[object]]:
    if window is None:
        return None, None
    try:
        import pygame  # type: ignore
        pygame.display.quit()
        pygame.quit()
    except ImportError:
        pass
    return None, None
