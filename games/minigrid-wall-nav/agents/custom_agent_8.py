"""
CustomAgent8 - RELATIVE COORDINATES (State Space Reduction)
===========================================================

PROBLEM WITH V7:
----------------
- State space: 10×10 × 4 directions × 10×10 = 40,000 states
- Each state visited only ~6 times during training
- Agent couldn't generalize "goal is ahead" across different positions
- Tabular Q-learning failed with sparse state coverage

SOLUTION: Relative Coordinates
------------------------------
OLD: (ax, ay, dir, gx, gy) - absolute positions
NEW: (dir, dx, dy)         - relative to agent

Where:
- dx = clamp(goal_x - agent_x, -2, 2)  # -2, -1, 0, 1, 2
- dy = clamp(goal_y - agent_y, -2, 2)  # -2, -1, 0, 1, 2
- dir = agent direction (0-3)

STATE SPACE MATH:
-----------------
4 directions × 5 dx values × 5 dy values = 100 states!

That's a 400x reduction from 40,000 states.

WHY THIS WORKS:
---------------
Instead of learning:
- "At (5,5) facing right with goal at (8,8), move forward"
- "At (3,3) facing right with goal at (6,6), move forward"

Agent learns ONE concept:
- "When goal is (+2, +2) relative and facing right, turn left then forward"

This generalizes to ANY position on the grid!

REWARD STRUCTURE (from v7):
---------------------------
- Dense reward: (prev_dist - curr_dist) × 0.1
- Step penalty: -0.01
- Wall bump: -0.1
- Timeout: -1.0
- Goal: +1.0

EXPLORATION (from v6):
----------------------
- Epsilon: 1.0 for first 1000 eps, then decay=0.998, min=0.05

EXPECTED RESULT:
----------------
With only 100 states and 5000 episodes:
- 5000 × 100 steps = 500k state visits
- 500k / 100 states = 5000 visits per state!
- Agent will thoroughly learn each situation
- Success rate should jump to 80%+
"""

import random 
import pickle
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, Any


class CustomAgent8:
    """
    Tabular Q-Learning Agent with RELATIVE COORDINATES.
    
    Key innovation: State is relative to agent position, not absolute.
    This reduces state space from 40,000 to ~100 states.
    
    State Representation:
        (agent_dir, dx, dy)
        
    Where:
        - agent_dir: 0=right, 1=down, 2=left, 3=up
        - dx: goal_x - agent_x, clamped to [-2, 2]
        - dy: goal_y - agent_y, clamped to [-2, 2]
    """

    def __init__(
        self,
        n_actions: int = 3,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.998,
        epsilon_min: float = 0.05,
        epsilon_delay_episodes: int = 1000,
        distance_reward_scale: float = 0.1,
        coord_clamp: int = 2,  # How far to track relative coords
    ):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon 
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_delay_episodes = epsilon_delay_episodes
        self.distance_reward_scale = distance_reward_scale
        self.coord_clamp = coord_clamp
        self.episode_count = 0
        
        # Q-table: MUCH smaller now!
        self.q_table: Dict[Tuple[Any, int], float] = defaultdict(float)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []

    def _clamp(self, value: int, min_val: int, max_val: int) -> int:
        """Clamp value to range [min_val, max_val]."""
        return max(min_val, min(max_val, value))

    def build_state(
        self,
        agent_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        agent_dir: int,
        dist_to_goal: float,      # Used for dense reward, not state
        dist_to_obstacle: float,  # Ignored
    ) -> Tuple:
        """
        Build RELATIVE state tuple.
        
        Args:
            agent_pos: (x, y) agent position
            goal_pos: (x, y) goal position
            agent_dir: Direction agent is facing (0=right, 1=down, 2=left, 3=up)
            dist_to_goal: Used for dense reward calculation
            dist_to_obstacle: Ignored (not needed with relative coords)
            
        Returns:
            State tuple: (agent_dir, dx, dy) where dx,dy are relative to agent
        """
        # Calculate relative position of goal
        dx = int(goal_pos[0]) - int(agent_pos[0])
        dy = int(goal_pos[1]) - int(agent_pos[1])
        
        # Clamp to reasonable range (goal far away = just head that direction)
        dx = self._clamp(dx, -self.coord_clamp, self.coord_clamp)
        dy = self._clamp(dy, -self.coord_clamp, self.coord_clamp)
        
        # State is just direction and relative offset
        # This is ONLY 4 × 5 × 5 = 100 unique states!
        state = (int(agent_dir), dx, dy)
        
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
        # Load coord_clamp if saved
        hyperparams = data.get("hyperparameters", {})
        self.coord_clamp = hyperparams.get("coord_clamp", 2)
        print(f"Q-table loaded from {filepath}")

    def reset_stats(self):
        """Reset training statistics."""
        self.episode_rewards = []
        self.episode_lengths = []
