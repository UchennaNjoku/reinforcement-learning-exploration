"""
Custom MiniGrid Environment: Random Goal & Wall Obstacle Navigation
====================================================================
10x10 playable grid where the agent, goal, and a wall obstacle are all randomly placed.
The wall is placed perpendicular to the agent-goal line, blocking the direct path.
The agent has full observability (knows its own position, the goal, and wall positions).

Author: Chenna (CS Senior, Bethune-Cookman University)
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv


class SimpleActionWrapper(gym.ActionWrapper):
    """
    Wrapper to restrict action space to only 3 useful navigation actions:
    0 = turn left, 1 = turn right, 2 = move forward
    
    This removes unused actions (pickup, drop, toggle, done) to speed up learning.
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(3)  # Only 3 actions
    
    def action(self, action):
        # Pass through unchanged - maps directly to MiniGrid's 0, 1, 2
        return action


class RandomGoalObstacleEnv(MiniGridEnv):
    """
    A MiniGrid environment where:
      - The agent start position is randomized
      - The goal position is randomized
      - A WALL obstacle is placed BETWEEN agent and goal, blocking direct path
      - The agent must navigate around the wall to reach the goal

    Observation (full observability):
      - agent_pos:       (x, y) coordinates of the agent
      - goal_pos:        (x, y) coordinates of the goal
      - obstacle_positions: List of (x, y) coordinates of wall tiles
      - dist_to_goal:    Euclidean distance from agent to goal

    Reward:
      - +1 (scaled by steps) for reaching the goal
      -  0 otherwise

    Actions (MiniGrid default):
      0 = turn left
      1 = turn right
      2 = move forward
      3 = pickup  (unused here)
      4 = drop    (unused here)
      5 = toggle  (unused here)
      6 = done    (unused here)
    """

    def __init__(
        self,
        size: int = 12,
        max_steps: int | None = None,
        wall_length: int = 4,
        wall_position_range: tuple[float, float] = (0.3, 0.7),
        **kwargs,
    ):
        """
        Args:
            size: Total grid size including walls. MiniGrid adds a 1-tile
                  border of walls, so size=12 gives a 10x10 playable area.
            max_steps: Maximum steps per episode before truncation.
            wall_length: Number of tiles in the wall obstacle (default 4).
            wall_position_range: Tuple of (min, max) fraction along agent-goal
                                 line where wall can be placed (default 0.3 to 0.7).
        """
        self.obstacle_positions = []
        self.goal_pos = None
        self.wall_length = wall_length
        self.wall_position_range = wall_position_range

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            highlight=False,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Navigate to the green goal while avoiding the wall blocking your path"

    def _is_reachable(self, start_pos, goal_pos, width, height):
        """
        BFS check: can we get from start_pos to goal_pos?
        Returns True if goal is reachable (ignoring other agents).
        """
        blocked = set(self.obstacle_positions)
        # Also consider border walls as blocked
        for x in range(width):
            for y in range(height):
                if x == 0 or x == width - 1 or y == 0 or y == height - 1:
                    blocked.add((x, y))
        
        queue = deque([tuple(start_pos)])
        visited = {tuple(start_pos)}
        
        while queue:
            x, y = queue.popleft()
            if (x, y) == tuple(goal_pos):
                return True
            
            # Check 4 neighbors (up, down, left, right)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 1 <= nx < width - 1 and 1 <= ny < height - 1:
                    if (nx, ny) not in blocked and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        
        return False

    def _gen_grid(self, width, height):
        """Generate the grid with wall obstacle placed between agent and goal."""
        max_retries = 100
        
        for attempt in range(max_retries):
            self.grid = Grid(width, height)
            self.grid.wall_rect(0, 0, width, height)
            self.obstacle_positions = []

            # Place goal at random position
            goal = Goal()
            self.place_obj(goal)
            self.goal_pos = (goal.cur_pos[0], goal.cur_pos[1])

            # Place agent at random position
            self.place_agent()
            agent_pos = np.array(self.agent_pos)
            goal_pos = np.array(self.goal_pos)

            # Place wall obstacle between agent and goal
            self.obstacle_positions = self._place_blocking_wall(
                agent_pos, goal_pos, width, height
            )
            
            # CRITICAL CHECKS:
            # 1. Goal must NOT be on a wall tile
            if self.goal_pos in self.obstacle_positions:
                if attempt < max_retries - 1:
                    continue  # Retry
                else:
                    print(f"WARNING: Goal on wall after {max_retries} retries")
            
            # 2. Goal must be reachable from agent start (BFS)
            if not self._is_reachable(agent_pos, goal_pos, width, height):
                if attempt < max_retries - 1:
                    continue  # Retry
                else:
                    print(f"WARNING: Goal unreachable after {max_retries} retries")
            
            # All checks passed
            break

    def _place_blocking_wall(self, agent_pos, goal_pos, width, height):
        """
        Place a wall perpendicular to the agent-goal line, somewhere between them.
        The wall position is randomized along the line.
        Returns list of wall tile positions.
        """
        t_min, t_max = self.wall_position_range
        t = np.random.uniform(t_min, t_max)

        wall_center = agent_pos + t * (goal_pos - agent_pos)
        wall_x, wall_y = int(wall_center[0]), int(wall_center[1])

        delta = goal_pos - agent_pos
        is_horizontal_path = abs(delta[0]) > abs(delta[1])

        wall_positions = []
        half_len = self.wall_length // 2

        # Try multiple positions along the line with jitter
        attempts = [
            (wall_x, wall_y),
            (wall_x + 1, wall_y),
            (wall_x - 1, wall_y),
            (wall_x, wall_y + 1),
            (wall_x, wall_y - 1),
        ]

        for attempt_x, attempt_y in attempts:
            if len(wall_positions) >= self.wall_length:
                break

            for offset in range(-half_len, half_len + 1):
                if is_horizontal_path:
                    wx, wy = attempt_x, attempt_y + offset
                else:
                    wx, wy = attempt_x + offset, attempt_y

                if 1 <= wx < width - 1 and 1 <= wy < height - 1:
                    if (wx, wy) != tuple(agent_pos) and (wx, wy) != tuple(goal_pos):
                        if self.grid.get(wx, wy) is None and (wx, wy) not in wall_positions:
                            self.grid.set(wx, wy, Wall())
                            wall_positions.append((wx, wy))

        return wall_positions

    def _get_custom_obs(self):
        """
        Build a custom observation dict with full board knowledge.
        """
        agent = np.array(self.agent_pos, dtype=np.float32)
        goal = np.array(self.goal_pos, dtype=np.float32)

        dist_to_goal = np.linalg.norm(agent - goal)

        if self.obstacle_positions:
            min_dist_to_obstacle = min(
                np.linalg.norm(agent - np.array(obs_pos))
                for obs_pos in self.obstacle_positions
            )
        else:
            min_dist_to_obstacle = 0.0

        return {
            "agent_pos": agent,
            "goal_pos": goal,
            "agent_dir": int(self.agent_dir),
            "obstacle_positions": self.obstacle_positions,
            "dist_to_goal": np.float32(dist_to_goal),
            "dist_to_obstacle": np.float32(min_dist_to_obstacle),
        }

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        info["custom_obs"] = self._get_custom_obs()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["custom_obs"] = self._get_custom_obs()
        return obs, reward, terminated, truncated, info

    def get_state_for_q_learning(self) -> tuple:
        """
        Convert current observation to a discrete state tuple for tabular Q-learning.
        Returns: (agent_x, agent_y, goal_x, goal_y)
        """
        return (
            int(self.agent_pos[0]),
            int(self.agent_pos[1]),
            int(self.goal_pos[0]),
            int(self.goal_pos[1]),
        )
