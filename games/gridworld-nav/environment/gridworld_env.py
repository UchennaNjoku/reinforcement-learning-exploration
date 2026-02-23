"""GridWorld Gymnasium environment (core dynamics only)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .constants import (
    UP, DOWN, LEFT, RIGHT,
    ACTION_DELTAS,
    TILE_EMPTY, TILE_WALL, TILE_GOAL,
    METADATA,
)
from .generation import generate_grid
from .rendering import render_frame, render_text, close_window


class GridWorldEnv(gym.Env):
    # constants
    UP = UP
    DOWN = DOWN
    LEFT = LEFT
    RIGHT = RIGHT
    ACTION_DELTAS = ACTION_DELTAS

    TILE_EMPTY = TILE_EMPTY
    TILE_WALL = TILE_WALL
    TILE_GOAL = TILE_GOAL

    metadata = METADATA

    def __init__(
        self,
        size: int = 12,
        max_steps: int | None = None,
        wall_length: int = 4,
        wall_position_range: tuple[float, float] = (0.3, 0.7),
        render_mode: str | None = None,
    ):
        self.size = int(size)
        self.playable_size = self.size - 2
        self.wall_length = int(wall_length)
        self.wall_position_range = wall_position_range
        self.render_mode = render_mode

        if max_steps is None:
            max_steps = 3 * self.size**2
        self.max_steps = int(max_steps)

        # Episode state
        self.steps: int = 0

        # Grid state
        self.grid: np.ndarray | None = None
        self.agent_pos: Tuple[int, int] | None = None
        self.goal_pos: Tuple[int, int] | None = None
        self.obstacle_positions: List[Tuple[int, int]] = []

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            "agent_pos": spaces.Box(low=1, high=self.size - 2, shape=(2,), dtype=np.int32),
            "goal_pos": spaces.Box(low=1, high=self.size - 2, shape=(2,), dtype=np.int32),
            "obstacle_positions": spaces.Sequence(
                spaces.Box(low=1, high=self.size - 2, shape=(2,), dtype=np.int32)
            ),
        })

        # for rendering
        self.cell_size = 50
        self._window = None
        self._clock = None

    # -- true if position is within playable area (not border)
    def _is_valid_pos(self, x: int, y: int) -> bool:
        return 1 <= x < self.size - 1 and 1 <= y < self.size - 1

    # -- true if position is blocked (wall or obstacle)
    def _is_blocked(self, x: int, y: int) -> bool:
        if not self._is_valid_pos(x, y):
            return True
        assert self.grid is not None
        return int(self.grid[y, x]) == TILE_WALL

    """
    Reset the environment to a new initial state.

    Returns:
        observation: Dict with agent_pos, goal_pos, obstacle_positions
        info: Dict with additional information
    """
    def reset(self, seed: int | None = None, options: dict | None = None
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        
        super().reset(seed=seed)

        # Use Gym's RNG when possible; fall back to numpy default RNG
        if seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(seed)

        self.steps = 0

        grid, agent_pos, goal_pos, obstacles = generate_grid(
            size=self.size,
            wall_length=self.wall_length,
            wall_position_range=self.wall_position_range,
            rng=rng,
        )

        self.grid = grid
        self.agent_pos = agent_pos
        self.goal_pos = goal_pos
        self.obstacle_positions = obstacles

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_human()

        return obs, info

    def _get_obs(self) -> Dict[str, Any]:
        assert self.agent_pos is not None and self.goal_pos is not None
        return {
            "agent_pos": np.array(self.agent_pos, dtype=np.int32),
            "goal_pos": np.array(self.goal_pos, dtype=np.int32),
            "obstacle_positions": [np.array(p, dtype=np.int32) for p in self.obstacle_positions],
        }

    def _get_info(self) -> Dict[str, Any]:
        assert self.agent_pos is not None and self.goal_pos is not None
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        dist_to_goal = float(np.sqrt((ax - gx) ** 2 + (ay - gy) ** 2))

        if self.obstacle_positions:
            min_dist_to_obstacle = min(
                float(np.sqrt((ax - ox) ** 2 + (ay - oy) ** 2))
                for ox, oy in self.obstacle_positions
            )
        else:
            min_dist_to_obstacle = 0.0

        return {
            "steps": int(self.steps),
            "dist_to_goal": dist_to_goal,
            "dist_to_obstacle": float(min_dist_to_obstacle),
        }

    def step(self, action: int) -> tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        assert self.grid is not None and self.agent_pos is not None and self.goal_pos is not None
        assert self.action_space.contains(action), f"Invalid action: {action}"

        self.steps += 1

        dx, dy = ACTION_DELTAS[int(action)]
        new_x = self.agent_pos[0] + dx
        new_y = self.agent_pos[1] + dy

        if not self._is_blocked(new_x, new_y):
            self.agent_pos = (new_x, new_y)

        terminated = self.agent_pos == self.goal_pos
        truncated = self.steps >= self.max_steps

        reward = 1.0 if terminated else 0.0

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_human()

        return obs, reward, terminated, truncated, info

    # -------------------- Rendering --------------------

    def render(self) -> np.ndarray | str | None:
        assert self.grid is not None and self.agent_pos is not None and self.goal_pos is not None

        if self.render_mode == "rgb_array":
            rgb, self._window, self._clock = render_frame(
                self.grid,
                self.agent_pos,
                self.goal_pos,
                cell_size=self.cell_size,
                render_mode="rgb_array",
                fps=self.metadata["render_fps"],
                window=self._window,
                clock=self._clock,
            )
            return rgb

        if self.render_mode == "ansi":
            return render_text(self.grid, self.agent_pos, self.goal_pos)

        if self.render_mode == "human":
            self._render_human()
            return None

        return None

    def _render_human(self) -> None:
        assert self.grid is not None and self.agent_pos is not None and self.goal_pos is not None
        _, self._window, self._clock = render_frame(
            self.grid,
            self.agent_pos,
            self.goal_pos,
            cell_size=self.cell_size,
            render_mode="human",
            fps=self.metadata["render_fps"],
            window=self._window,
            clock=self._clock,
        )

    def close(self):
        self._window, self._clock = close_window(self._window, self._clock)
