"""
CustomAgent5 - SIMPLIFIED STATE SPACE
======================================

PROBLEM WITH V4:
----------------
- 272k states in Q-table!
- Each state visited only ~4 times during training
- Not enough data to learn good policy
- Agent memorized instead of generalized

SOLUTION: Simplify state representation
---------------------------------------
OLD: (ax, ay, dir, gx, gy, dist_goal_rounded, dist_obs_rounded) = 272k states
NEW: (ax, ay, dir, gx, gy) = much smaller!

Why remove distances?
- Distances are DERIVABLE from positions
- Having both is redundant
- Fewer states = more visits per state = better learning

EXPECTED STATE SPACE:
- 10x10 agent positions × 4 directions × 10x10 goal positions = 40,000 states max
- Actually much less (agent and goal can't be on same tile, borders, etc.)
- More realistic: ~10k states = 100 visits each during training

REWARD STRUCTURE (simplified):
------------------------------
| Event           | Reward |
|-----------------|--------|
| Per step        | -0.01  |
| Wall bump       | -0.1   |
| Reach goal      | +1.0   |
| Timeout         | -1.0   |

(Removed forward/turn bonuses - just keep it simple)
"""

import random 
import pickle
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, Any


class CustomAgent5:
    """
    Tabular Q-Learning Agent with MINIMAL state representation.
    
    State Representation:
        (agent_x, agent_y, agent_dir, goal_x, goal_y)
        
    Only 5 features instead of 7 - removes redundant distance features.
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
        # State: (ax, ay, dir, gx, gy) - much simpler!
        self.q_table: Dict[Tuple[Any, int], float] = defaultdict(float)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []

    def build_state(
        self,
        agent_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        agent_dir: int,
        dist_to_goal: float,      # Ignored - kept for API compatibility
        dist_to_obstacle: float,  # Ignored - kept for API compatibility
    ) -> Tuple:
        """
        Build the MINIMAL state tuple.
        
        Args:
            agent_pos: (x, y) agent position
            goal_pos: (x, y) goal position  
            agent_dir: Direction agent is facing (0=right, 1=down, 2=left, 3=up)
            dist_to_goal: IGNORED (can be calculated from positions)
            dist_to_obstacle: IGNORED (not needed for navigation)
            
        Returns:
            State tuple: (ax, ay, dir, gx, gy)
        """
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
