"""Constants for the GridWorld navigation environment."""

from __future__ import annotations

from typing import Dict, Tuple

# Actions
UP: int = 0
DOWN: int = 1
LEFT: int = 2
RIGHT: int = 3

ACTION_DELTAS: Dict[int, Tuple[int, int]] = {
    UP: (0, -1),
    DOWN: (0, 1),
    LEFT: (-1, 0),
    RIGHT: (1, 0),
}

# Tiles (grid[y, x])
TILE_EMPTY: int = 0
TILE_WALL: int = 1
TILE_GOAL: int = 2

METADATA = {
    "render_modes": ["human", "rgb_array", "ansi"],
    "render_fps": 10,
}
