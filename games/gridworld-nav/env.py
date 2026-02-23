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

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from typing import Tuple, List, Dict, Any, Optional


class GridWorldEnv(gym.Env):
    """
    A GridWorld environment where:
      - The agent start position is randomized
      - The goal position is randomized
      - A WALL obstacle is placed BETWEEN agent and goal, blocking direct path
      - The agent must navigate around the wall to reach the goal
      - Agent moves in cardinal directions (no orientation)

    Observation (full observability):
      - agent_pos:       (x, y) coordinates of the agent
      - goal_pos:        (x, y) coordinates of the goal
      - obstacle_positions: List of (x, y) coordinates of wall tiles

    Actions (Discrete(4)):
      0 = UP    (y - 1)
      1 = DOWN  (y + 1)
      2 = LEFT  (x - 1)
      3 = RIGHT (x + 1)
    """

    # Action constants
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    # Movement deltas: (dx, dy)
    ACTION_DELTAS = {
        UP: (0, -1),
        DOWN: (0, 1),
        LEFT: (-1, 0),
        RIGHT: (1, 0),
    }

    # Tile types for rendering
    TILE_EMPTY = 0
    TILE_WALL = 1
    TILE_GOAL = 2
    TILE_AGENT = 3

    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 10,
    }

    def __init__(
        self,
        size: int = 12,
        max_steps: int | None = None,
        wall_length: int = 4,
        wall_position_range: tuple[float, float] = (0.3, 0.7),
        render_mode: str | None = None,
    ):
        """
        Args:
            size: Total grid size including walls. Border of 1-tile walls,
                  so size=12 gives a 10x10 playable area.
            max_steps: Maximum steps per episode before truncation.
            wall_length: Number of tiles in the wall obstacle (default 4).
            wall_position_range: Tuple of (min, max) fraction along agent-goal
                                 line where wall can be placed (default 0.3 to 0.7).
            render_mode: Rendering mode ("human", "rgb_array", "ansi", or None).
        """
        self.size = size
        self.playable_size = size - 2  # Subtract border walls
        self.wall_length = wall_length
        self.wall_position_range = wall_position_range
        self.render_mode = render_mode

        if max_steps is None:
            max_steps = 3 * size**2
        self.max_steps = max_steps

        # Grid state
        self.grid: np.ndarray | None = None  # 2D array of tile types
        self.agent_pos: Tuple[int, int] | None = None
        self.goal_pos: Tuple[int, int] | None = None
        self.obstacle_positions: List[Tuple[int, int]] = []

        # Episode state
        self.steps: int = 0

        # Action space: 4 cardinal directions
        self.action_space = spaces.Discrete(4)

        # Observation space
        # We use a Dict space for structured observations
        self.observation_space = spaces.Dict({
            "agent_pos": spaces.Box(
                low=1, high=size - 2,
                shape=(2,), dtype=np.int32
            ),
            "goal_pos": spaces.Box(
                low=1, high=size - 2,
                shape=(2,), dtype=np.int32
            ),
            "obstacle_positions": spaces.Sequence(
                spaces.Box(low=1, high=size - 2, shape=(2,), dtype=np.int32)
            ),
        })

        # For rendering
        self.window = None
        self.clock = None
        self.cell_size = 50  # pixels per cell

    def _is_valid_pos(self, x: int, y: int) -> bool:
        """Check if position is within playable area (not border)."""
        return 1 <= x < self.size - 1 and 1 <= y < self.size - 1

    def _is_blocked(self, x: int, y: int) -> bool:
        """Check if position is blocked (wall or obstacle)."""
        if not self._is_valid_pos(x, y):
            return True
        if self.grid is None:
            return False
        return self.grid[y, x] == self.TILE_WALL

    def _is_reachable(self, start_pos: Tuple[int, int], goal_pos: Tuple[int, int]) -> bool:
        """
        BFS check: can we get from start_pos to goal_pos?
        Returns True if goal is reachable.
        """
        queue = deque([start_pos])
        visited = {start_pos}

        while queue:
            x, y = queue.popleft()
            if (x, y) == goal_pos:
                return True

            # Check 4 neighbors (up, down, left, right)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if self._is_valid_pos(nx, ny):
                    if not self._is_blocked(nx, ny) and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))

        return False

    def _place_blocking_wall(
        self,
        agent_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """
        Place a wall roughly perpendicular to the agent-goal line somewhere between them.
        Returns exactly self.wall_length placed tiles (unless impossible at all attempts).

        Notes:
        - Coordinate convention: grid[y, x]
        - If delta_x > delta_y (goal mostly left/right), we place a VERTICAL wall (x fixed).
        Else we place a HORIZONTAL wall (y fixed).
        """
        t_min, t_max = self.wall_position_range
        t = np.random.uniform(t_min, t_max)

        ax, ay = agent_pos
        gx, gy = goal_pos

        # Pick an integer "center" cell along the segment agent->goal
        wall_x = int(round(ax + t * (gx - ax)))
        wall_y = int(round(ay + t * (gy - ay)))

        delta_x = abs(gx - ax)
        delta_y = abs(gy - ay)
        path_is_more_horizontal = delta_x > delta_y

        # Offsets with EXACT length = wall_length, centered around 0
        # Example L=4 => offsets [-1, 0, 1, 2] (or [-2,-1,0,1] also fine; this keeps symmetry-ish)
        # Example L=5 => offsets [-2,-1,0,1,2]
        L = int(self.wall_length)
        start = -(L // 2)
        offsets = list(range(start, start + L))  # length L always

        # Candidate centers to try (jitter)
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
                    # VERTICAL wall: x fixed, vary y
                    wx, wy = cx, cy + off
                else:
                    # HORIZONTAL wall: y fixed, vary x
                    wx, wy = cx + off, cy

                if not self._is_valid_pos(wx, wy):
                    continue
                if (wx, wy) == agent_pos or (wx, wy) == goal_pos:
                    continue
                if self.grid[wy, wx] != self.TILE_EMPTY:
                    continue

                self.grid[wy, wx] = self.TILE_WALL
                placed.append((wx, wy))

                if len(placed) == L:
                    return placed  # EXACT length reached

            # If we didn't manage to place L tiles, undo partial placement and try next jitter center
            for (wx, wy) in placed:
                self.grid[wy, wx] = self.TILE_EMPTY

        # If all attempts fail, return empty and let generator retry
        return []

    def _generate_grid(self) -> bool:
        """
        Generate the grid with wall obstacle placed between agent and goal.
        Returns True on success, False if generation failed after retries.
        """
        max_retries = 100

        for attempt in range(max_retries):
            # Initialize grid with empty tiles
            self.grid = np.full((self.size, self.size), self.TILE_EMPTY, dtype=np.int32)

            # Place border walls
            self.grid[0, :] = self.TILE_WALL
            self.grid[-1, :] = self.TILE_WALL
            self.grid[:, 0] = self.TILE_WALL
            self.grid[:, -1] = self.TILE_WALL

            self.obstacle_positions = []

            # Randomly place goal (within playable area)
            gx = np.random.randint(1, self.size - 1)
            gy = np.random.randint(1, self.size - 1)
            self.goal_pos = (gx, gy)
            self.grid[gy, gx] = self.TILE_GOAL

            # Randomly place agent (different from goal)
            while True:
                ax = np.random.randint(1, self.size - 1)
                ay = np.random.randint(1, self.size - 1)
                if (ax, ay) != self.goal_pos:
                    self.agent_pos = (ax, ay)
                    break

            # Place wall obstacle between agent and goal
            self.obstacle_positions = self._place_blocking_wall(
                self.agent_pos, self.goal_pos
            )

            # CRITICAL CHECKS:
            # 1. Goal must NOT be on a wall tile
            if self.goal_pos in self.obstacle_positions:
                if attempt < max_retries - 1:
                    continue  # Retry
                else:
                    print(f"WARNING: Goal on wall after {max_retries} retries")

            # 2. Goal must be reachable from agent start (BFS)
            if not self._is_reachable(self.agent_pos, self.goal_pos):
                if attempt < max_retries - 1:
                    continue  # Retry
                else:
                    print(f"WARNING: Goal unreachable after {max_retries} retries")
                    return False

            # All checks passed
            return True

        return False

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment to a new initial state.

        Returns:
            observation: Dict with agent_pos, goal_pos, obstacle_positions
            info: Dict with additional information
        """
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        self.steps = 0

        # Generate grid (with retries for reachability)
        success = self._generate_grid()
        if not success:
            # Last resort: create a simple solvable configuration
            self._create_fallback_grid()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _create_fallback_grid(self):
        """Create a simple guaranteed-solvable grid as fallback."""
        self.grid = np.full((self.size, self.size), self.TILE_EMPTY, dtype=np.int32)
        self.grid[0, :] = self.TILE_WALL
        self.grid[-1, :] = self.TILE_WALL
        self.grid[:, 0] = self.TILE_WALL
        self.grid[:, -1] = self.TILE_WALL

        # Agent at bottom-left, goal at top-right
        self.agent_pos = (1, self.size - 2)
        self.goal_pos = (self.size - 2, 1)
        self.grid[self.goal_pos[1], self.goal_pos[0]] = self.TILE_GOAL
        self.obstacle_positions = []

    def _get_obs(self) -> Dict[str, Any]:
        """Build the observation dict."""
        return {
            "agent_pos": np.array(self.agent_pos, dtype=np.int32),
            "goal_pos": np.array(self.goal_pos, dtype=np.int32),
            "obstacle_positions": [
                np.array(pos, dtype=np.int32) for pos in self.obstacle_positions
            ],
        }

    def _get_info(self) -> Dict[str, Any]:
        """Build the info dict with additional information."""
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        dist_to_goal = np.sqrt((ax - gx) ** 2 + (ay - gy) ** 2)

        # Calculate minimum distance to any obstacle
        if self.obstacle_positions:
            min_dist_to_obstacle = min(
                np.sqrt((ax - ox) ** 2 + (ay - oy) ** 2)
                for ox, oy in self.obstacle_positions
            )
        else:
            min_dist_to_obstacle = 0.0

        return {
            "steps": self.steps,
            "dist_to_goal": float(dist_to_goal),
            "dist_to_obstacle": float(min_dist_to_obstacle),
        }

    def step(self, action: int) -> tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute one timestep within the environment.

        Args:
            action: Integer in [0, 3] representing UP/DOWN/LEFT/RIGHT

        Returns:
            observation: Dict with agent_pos, goal_pos, obstacle_positions
            reward: Float reward value
            terminated: Bool whether episode ended (goal reached)
            truncated: Bool whether episode was truncated (max steps)
            info: Dict with additional information
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        self.steps += 1

        # Calculate new position based on action
        dx, dy = self.ACTION_DELTAS[action]
        new_x = self.agent_pos[0] + dx
        new_y = self.agent_pos[1] + dy

        # Check if move is valid (not into a wall)
        if not self._is_blocked(new_x, new_y):
            # Update agent position
            self.agent_pos = (new_x, new_y)

        # Check for goal reached
        terminated = self.agent_pos == self.goal_pos

        # Check for truncation (max steps)
        truncated = self.steps >= self.max_steps

        # Calculate reward
        if terminated:
            reward = 1.0
        else:
            reward = 0.0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray | str | None:
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "ansi":
            return self._render_text()
        elif self.render_mode == "human":
            self._render_frame()

    def _render_frame(self) -> np.ndarray:
        """Render the current frame as RGB array using pygame."""
        try:
            import pygame
        except ImportError:
            print("pygame not installed. Install with: pip install pygame")
            return np.zeros((self.size * self.cell_size, self.size * self.cell_size, 3), dtype=np.uint8)

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.size * self.cell_size, self.size * self.cell_size)
            )
            pygame.display.set_caption("GridWorld Navigation")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Create canvas
        canvas = pygame.Surface((self.size * self.cell_size, self.size * self.cell_size))
        canvas.fill((200, 200, 200))  # Background gray

        # Colors
        COLOR_WALL = (64, 64, 64)
        COLOR_GOAL = (0, 200, 0)
        COLOR_AGENT = (0, 0, 200)
        COLOR_EMPTY = (240, 240, 240)
        COLOR_GRID = (180, 180, 180)

        # Draw grid
        for y in range(self.size):
            for x in range(self.size):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )

                tile = self.grid[y, x]
                if tile == self.TILE_WALL:
                    pygame.draw.rect(canvas, COLOR_WALL, rect)
                elif tile == self.TILE_GOAL:
                    pygame.draw.rect(canvas, COLOR_GOAL, rect)
                else:
                    pygame.draw.rect(canvas, COLOR_EMPTY, rect)

                # Grid lines
                pygame.draw.rect(canvas, COLOR_GRID, rect, 1)

        # Draw agent (as a circle)
        ax, ay = self.agent_pos
        center = (
            ax * self.cell_size + self.cell_size // 2,
            ay * self.cell_size + self.cell_size // 2,
        )
        radius = self.cell_size // 3
        pygame.draw.circle(canvas, COLOR_AGENT, center, radius)

        if self.render_mode == "human":
            # Copy canvas to window
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:
            # Return RGB array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def _render_text(self) -> str:
        """Render the current state as ASCII text."""
        lines = []
        for y in range(self.size):
            row = ""
            for x in range(self.size):
                if (x, y) == self.agent_pos:
                    row += " A"
                elif (x, y) == self.goal_pos:
                    row += " G"
                elif self.grid[y, x] == self.TILE_WALL:
                    row += " #"
                else:
                    row += " ."
            lines.append(row)
        return "\n".join(lines)

    def close(self):
        """Clean up resources."""
        if self.window is not None:
            try:
                import pygame
                pygame.display.quit()
                pygame.quit()
            except ImportError:
                pass
            self.window = None
            self.clock = None

    
