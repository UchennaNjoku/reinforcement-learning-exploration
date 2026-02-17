"""
CustomAgent7 - DENSE REWARD SHAPING
====================================

PROBLEM WITH V6:
----------------
- Only +1 reward at goal (very sparse)
- Agent wanders 200 steps with no feedback
- By episode 9000, still not reliably finding goal
- No signal whether actions help or hurt progress

SOLUTION: Dense reward based on distance to goal
------------------------------------------------
Reward the agent EVERY STEP based on progress:
- Getting CLOSER to goal: positive reward
- Getting FARTHER from goal: negative reward

REWARD CALCULATION:
-------------------
reward = (prev_dist - curr_dist) * scale_factor

Examples:
- Was 10 away, now 8 away: (10-8) * 0.1 = +0.2 (good progress!)
- Was 5 away, now 6 away: (5-6) * 0.1 = -0.1 (wrong direction!)
- Reached goal (dist 0): includes +1.0 base reward

This is called "POTENTIAL-BASED REWARD SHAPING" and it preserves
the optimal policy while speeding up learning dramatically.

OTHER FIXES (from v6):
----------------------
- Epsilon: 1.0 for first 1000 eps, then decay=0.998, min=0.05
- State: (ax, ay, dir, gx, gy) - minimal
- Wall bump: -0.1
- Step penalty: -0.01
- Timeout: -1.0

EXPECTED BEHAVIOR:
------------------
- Agent learns within first few hundred episodes
- Every step provides learning signal
- Natural gradient toward goal emerges
- Success rate should jump dramatically
"""

import random 
import pickle
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, Any


class CustomAgent7:
    """
    Tabular Q-Learning Agent with DENSE REWARD SHAPING.
    
    Key insight: Reward progress toward goal every single step.
    
    State Representation:
        (agent_x, agent_y, agent_dir, goal_x, goal_y)
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
        distance_reward_scale: float = 0.1,  # Scale for dense reward
    ):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon 
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_delay_episodes = epsilon_delay_episodes
        self.distance_reward_scale = distance_reward_scale
        self.episode_count = 0
        
        # Q-table
        self.q_table: Dict[Tuple[Any, int], float] = defaultdict(float)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []

    def build_state(
        self,
        agent_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        agent_dir: int,
        dist_to_goal: float,      # Used for dense reward
        dist_to_obstacle: float,  # Ignored
    ) -> Tuple:
        """Build the minimal state tuple."""
        state = (
            int(agent_pos[0]),
            int(agent_pos[1]),
            int(agent_dir),
            int(goal_pos[0]),
            int(goal_pos[1]),
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
