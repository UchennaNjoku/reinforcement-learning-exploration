"""
CustomAgent10 - TRI-DIRECTIONAL WALL SENSORS
============================================

PROBLEM WITH V9:
----------------
- Had nearest wall vector (dxw, dyw) but didn't know if LEFT or RIGHT was blocked
- Agent would try to turn into walls
- Wasted steps testing directions

SOLUTION: 3 Binary Wall Sensors
-------------------------------
Replace (dxw, dyw) with:
- front_blocked: 1 if wall directly ahead, else 0
- left_blocked:  1 if wall to the left, else 0  
- right_blocked: 1 if wall to the right, else 0

This gives the agent complete awareness of its immediate surroundings
in the three directions that matter for navigation.

STATE REPRESENTATION:
---------------------
(dir, dxg, dyg, front_blocked, left_blocked, right_blocked)
- dir: 4 values (0-3)
- dxg, dyg: 9 values each (±4 clamp)
- front_blocked: 2 values
- left_blocked: 2 values  
- right_blocked: 2 values

Total: 4 × 9 × 9 × 2 × 2 × 2 = 2,592 states
(Much smaller than v9's 52k states!)

EPSILON_MIN REDUCED (0.05 -> 0.01):
-----------------------------------
More exploitation, less random disruption during evaluation.

OTHER FEATURES (from v9):
-------------------------
- learning_rate: 0.5 (fast)
- epsilon_decay: 0.9995 (slow)
- epsilon_delay: 2000 episodes
- coord_clamp: 4
- Dense reward based on distance improvement
- Step penalty: -0.01
- Wall bump penalty: -0.05
- Timeout penalty: -1.0
"""

import random 
import pickle
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, Any, List


class CustomAgent10:
    """
    Tabular Q-Learning Agent with tri-directional wall sensors.
    
    Key innovation: Knows about walls in front, left, AND right directions.
    
    State Representation:
        (dir, dxg, dyg, front_blocked, left_blocked, right_blocked)
    """

    def __init__(
        self,
        n_actions: int = 3,
        learning_rate: float = 0.5,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.9995,
        epsilon_min: float = 0.01,  # REDUCED from 0.05
        epsilon_delay_episodes: int = 2000,
        epsilon_delay: int = None,
        distance_reward_scale: float = 0.1,
        coord_clamp: int = 4,
    ):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon 
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_delay_episodes = epsilon_delay if epsilon_delay is not None else epsilon_delay_episodes
        self.distance_reward_scale = distance_reward_scale
        self.coord_clamp = coord_clamp
        self.episode_count = 0
        
        self.q_table: Dict[Tuple[Any, int], float] = defaultdict(float)
        
        self.episode_rewards = []
        self.episode_lengths = []

    def _clamp(self, value: int, min_val: int, max_val: int) -> int:
        """Clamp value to range [min_val, max_val]."""
        return max(min_val, min(max_val, value))

    def _get_cell_in_direction(self, agent_pos: Tuple[int, int], direction: int) -> Tuple[int, int]:
        """
        Get the cell in a specific direction from agent.
        
        direction: 0=right, 1=down, 2=left, 3=up
        Returns: (x, y) coordinates
        """
        ax, ay = int(agent_pos[0]), int(agent_pos[1])
        
        if direction == 0:    # right
            return (ax + 1, ay)
        elif direction == 1:  # down
            return (ax, ay + 1)
        elif direction == 2:  # left
            return (ax - 1, ay)
        else:                 # up (dir == 3)
            return (ax, ay - 1)
    
    def _is_blocked(self, cell: Tuple[int, int], obstacle_positions: List) -> int:
        """
        Check if a cell is blocked (out of bounds or is an obstacle).
        
        Returns: 1 if blocked, 0 if free
        """
        cx, cy = int(cell[0]), int(cell[1])
        
        # Check bounds (playable area is 1-10 for a 12x12 grid with walls on border)
        if cx < 1 or cx > 10 or cy < 1 or cy > 10:
            return 1  # Out of bounds = blocked
        
        # Check if cell is an obstacle
        if obstacle_positions:
            for ox, oy in obstacle_positions:
                if int(ox) == cx and int(oy) == cy:
                    return 1  # Is an obstacle
        
        return 0  # Free to move
    
    def _get_relative_direction(self, agent_dir: int, offset: int) -> int:
        """
        Get direction relative to agent's facing direction.
        
        agent_dir: current facing (0=right, 1=down, 2=left, 3=up)
        offset: 0=front, -1=left (turn left), +1=right (turn right)
        
        Returns: absolute direction (0-3)
        """
        return (agent_dir + offset) % 4

    def build_state(
        self,
        agent_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        agent_dir: int,
        dist_to_goal: float,
        dist_to_obstacle: float,
        obstacle_positions=None,
    ) -> Tuple:
        """
        Build state with goal relative position AND tri-directional wall sensors.
        
        State: (dir, dxg, dyg, front_blocked, left_blocked, right_blocked)
        - dir: agent direction (0-3)
        - dxg, dyg: goal position relative to agent (clamped)
        - front_blocked: 1 if wall directly ahead, else 0
        - left_blocked: 1 if wall to left of agent, else 0
        - right_blocked: 1 if wall to right of agent, else 0
        """
        ax, ay = int(agent_pos[0]), int(agent_pos[1])
        gx, gy = int(goal_pos[0]), int(goal_pos[1])
        
        # Goal relative position
        dxg = self._clamp(gx - ax, -self.coord_clamp, self.coord_clamp)
        dyg = self._clamp(gy - ay, -self.coord_clamp, self.coord_clamp)
        
        # TRI-DIRECTIONAL WALL SENSORS
        # Check front (direction agent is facing)
        front_dir = agent_dir
        front_cell = self._get_cell_in_direction(agent_pos, front_dir)
        front_blocked = self._is_blocked(front_cell, obstacle_positions)
        
        # Check left (turn left from current facing)
        left_dir = self._get_relative_direction(agent_dir, -1)  # -1 = turn left
        left_cell = self._get_cell_in_direction(agent_pos, left_dir)
        left_blocked = self._is_blocked(left_cell, obstacle_positions)
        
        # Check right (turn right from current facing)
        right_dir = self._get_relative_direction(agent_dir, 1)  # +1 = turn right
        right_cell = self._get_cell_in_direction(agent_pos, right_dir)
        right_blocked = self._is_blocked(right_cell, obstacle_positions)
        
        # State: direction, goal relative, and 3 wall sensors
        # State size: 4 × 9 × 9 × 2 × 2 × 2 = 2,592 states
        state = (int(agent_dir), dxg, dyg, front_blocked, left_blocked, right_blocked)
        return state

    def get_action(self, state: Tuple, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return self._get_best_action(state)

    def _get_best_action(self, state: Tuple) -> int:
        """Get the action with highest Q-value for given state."""
        q_values = [self.q_table[(state, a)] for a in range(self.n_actions)]
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return int(np.random.choice(best_actions))

    def update(
        self,
        state: Tuple,
        action: int,
        reward: float,
        next_state: Tuple,
        terminated: bool,
    ) -> float:
        """Update Q-value using the Q-learning update rule."""
        current_q = self.q_table[(state, action)]
        
        if terminated:
            target_q = reward
        else:
            next_q_values = [self.q_table[(next_state, a)] for a in range(self.n_actions)]
            max_next_q = max(next_q_values)
            target_q = reward + self.discount_factor * max_next_q
        
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[(state, action)] = new_q
        
        return abs(new_q - current_q)

    def decay_epsilon(self):
        """Decay epsilon with delay."""
        self.episode_count += 1
        if self.episode_count > self.epsilon_delay_episodes:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_q_table_size(self) -> int:
        """Return number of state-action pairs in Q-table."""
        return len(self.q_table)

    def get_state_count(self) -> int:
        """Return number of unique states visited."""
        return len(self.q_table) // self.n_actions

    def save(self, filepath: str):
        """Save the Q-table to a file."""
        data = {
            "q_table": dict(self.q_table),
            "epsilon": self.epsilon,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "hyperparameters": {
                "n_actions": self.n_actions,
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "epsilon_decay": self.epsilon_decay,
                "epsilon_min": self.epsilon_min,
                "epsilon_delay_episodes": self.epsilon_delay_episodes,
                "distance_reward_scale": self.distance_reward_scale,
                "coord_clamp": self.coord_clamp,
            },
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"Q-table saved to {filepath}")

    def load(self, filepath: str):
        """Load the Q-table from a file."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(float, data["q_table"])
        self.epsilon = data["epsilon"]
        self.episode_rewards = data.get("episode_rewards", [])
        self.episode_lengths = data.get("episode_lengths", [])
        hyperparams = data.get("hyperparameters", {})
        self.coord_clamp = hyperparams.get("coord_clamp", 4)
        self.epsilon_min = hyperparams.get("epsilon_min", 0.01)
        print(f"Q-table loaded from {filepath}")

    def reset_stats(self):
        """Reset training statistics."""
        self.episode_rewards = []
        self.episode_lengths = []
