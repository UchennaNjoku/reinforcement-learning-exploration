"""
CustomAgent3 - FIXED VERSION
============================

FIXES APPLIED (based on CustomAgent2 failures):
------------------------------------------------
1. ADDED STEP PENALTY (-0.08 per step)
   - Encourages agent to finish quickly
   - Wandering is now costly
   
2. ADDED TIMEOUT PENALTY (-1.0)
   - Hitting max_steps is as bad as failing
   - Strong signal to avoid timeouts
   
3. REDUCED ACTION REWARD SHAPING
   - Forward bonus: +0.05 -> +0.02 (reduced by 60%)
   - Turn penalty: -0.02 -> -0.01 (reduced by 50%)
   - Less likely to overwhelm goal reward
   
4. STATE SPACE (unchanged but monitored)
   - Still: (ax, ay, dir, gx, gy, dist_goal, dist_obs)
   - Watch for state explosion in training logs

REWARD STRUCTURE:
-----------------
| Event                | Reward |
|----------------------|--------|
| Per step             | -0.08  |
| Move forward         | +0.02  |
| Turn (left/right)    | -0.01  |
| Reach goal           | +1.0   |
| Timeout (max_steps)  | -1.0   |

GOAL: Agent should learn to reach goal efficiently rather than wander.
"""

import random 
import pickle
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, Any


class CustomAgent3:
    """
    Tabular Q-Learning Agent with improved reward handling.
    
    State Representation:
        (agent_x, agent_y, agent_dir, goal_x, goal_y, dist_to_goal_rounded, dist_to_obstacle_rounded)
        
    The agent_dir feature (0=right, 1=down, 2=left, 3=up) is critical because
    actions are relative (turn_left, turn_right, move_forward).
    """

    def __init__(
        self,
        n_actions: int = 3,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon 
        self.epsilon_decay = epsilon_decay 
        self.epsilon_min = epsilon_min 

        # Q-table: dictionary mapping (state_tuple, action) -> Q-value
        self.q_table: Dict[Tuple[Any, int], float] = defaultdict(float)
        
        # Distance precision settings
        self.goal_dist_precision = 1.0
        self.obstacle_dist_precision = 0.5
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []

    def _round_distance(self, distance: float, precision: float) -> float:
        """Round distance to specified precision for state discretization."""
        return round(distance / precision) * precision

    def build_state(
        self,
        agent_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        agent_dir: int,
        dist_to_goal: float,
        dist_to_obstacle: float
    ) -> Tuple:
        """
        Build the state tuple from environment observations.
        
        Args:
            agent_pos: (x, y) agent position
            goal_pos: (x, y) goal position
            agent_dir: Direction agent is facing (0=right, 1=down, 2=left, 3=up)
            dist_to_goal: Euclidean distance to goal (continuous)
            dist_to_obstacle: Min distance to any wall obstacle (continuous)
            
        Returns:
            State tuple: (ax, ay, dir, gx, gy, dist_goal_rounded, dist_obs_rounded)
        """
        goal_dist_rounded = self._round_distance(dist_to_goal, self.goal_dist_precision)
        obs_dist_rounded = self._round_distance(dist_to_obstacle, self.obstacle_dist_precision)
        
        state = (
            int(agent_pos[0]),
            int(agent_pos[1]),
            int(agent_dir),
            int(goal_pos[0]),
            int(goal_pos[1]),
            goal_dist_rounded,
            obs_dist_rounded,
        )
        
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
        """Decay epsilon after each episode."""
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
        print(f"Q-table loaded from {filepath}")

    def reset_stats(self):
        """Reset training statistics."""
        self.episode_rewards = []
        self.episode_lengths = []
