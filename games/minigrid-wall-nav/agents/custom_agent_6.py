"""
CustomAgent6 - IMPROVED EXPLORATION
====================================

PROBLEM WITH V4/V5:
-------------------
- Epsilon decayed too fast (0.995 per episode)
- By episode 900, epsilon = 0.01 (99% exploitation)
- Agent stopped exploring before learning the environment
- Only 4% success rate - agent exploiting incomplete knowledge

EXPLORATION MATH (old vs new):
------------------------------
Episode    | v5 (0.995) | v6 (0.998, delayed)
-----------|------------|--------------------
0          | 1.00       | 1.00 (no decay)
100        | 0.60       | 1.00 (no decay)
500        | 0.08       | 1.00 (no decay)
1000       | 0.01       | 1.00 -> 0.998 (start decay)
2000       | 0.01       | 0.82
3000       | 0.01       | 0.67
4000       | 0.01       | 0.55
5000       | 0.01       | 0.45
10000      | 0.01       | 0.20

SOLUTION: Three-pronged approach
--------------------------------
1. SLOWER DECAY (0.995 -> 0.998)
   - Takes longer to reach minimum
   - More exploration throughout training
   
2. HIGHER MINIMUM (0.01 -> 0.05)
   - Always keeps 5% exploration
   - Prevents getting stuck in local optima
   
3. DELAYED DECAY (first 1000 episodes at 1.0)
   - Pure exploration phase
   - Discover the environment completely
   - Build comprehensive Q-table before exploiting

REWARD STRUCTURE (same as v5):
------------------------------
| Event           | Reward |
|-----------------|--------|
| Per step        | -0.01  |
| Wall bump       | -0.1   |
| Reach goal      | +1.0   |
| Timeout         | -1.0   |

STATE REPRESENTATION (same as v5):
-----------------------------------
(ax, ay, dir, gx, gy) - minimal, no redundant distances

EXPECTED IMPROVEMENT:
---------------------
- First 1000 eps: Pure exploration, finds many paths
- Eps 1000-5000: Gradual shift to exploitation
- After 5000: Still 20-50% exploration, keeps discovering
- Should dramatically improve success rate
"""

import random 
import pickle
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, Any


class CustomAgent6:
    """
    Tabular Q-Learning Agent with IMPROVED EXPLORATION.
    
    Key improvements:
    - Slower epsilon decay (0.998 vs 0.995)
    - Higher minimum epsilon (0.05 vs 0.01)
    - Delayed decay (first 1000 episodes at 1.0)
    
    State Representation:
        (agent_x, agent_y, agent_dir, goal_x, goal_y)
    """

    def __init__(
        self,
        n_actions: int = 3,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.998,    # SLOWER decay (was 0.995)
        epsilon_min: float = 0.05,        # HIGHER minimum (was 0.01)
        epsilon_delay_episodes: int = 1000,  # NEW: Don't decay for first 1000 eps
    ):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon 
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_delay_episodes = epsilon_delay_episodes  # NEW
        self.episode_count = 0  # Track episodes for delay
        
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
        dist_to_goal: float,      # Ignored
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
        """
        Decay epsilon with DELAY.
        First 1000 episodes: stay at 1.0 (pure exploration)
        After: decay slowly at 0.998 rate
        """
        self.episode_count += 1
        
        # Only start decaying after delay period
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
