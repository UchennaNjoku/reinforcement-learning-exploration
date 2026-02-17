"""
Tabular Q-Learning Agent
========================
A simple Q-learning implementation using a dictionary-based Q-table.

Q-learning update rule:
    Q(s, a) = Q(s, a) + alpha * [reward + gamma * max(Q(s', a')) - Q(s, a)]
"""

from __future__ import annotations

import numpy as np
import pickle
from collections import defaultdict
from typing import Dict, Tuple, Any


class QLearningAgent:
    """
    Tabular Q-Learning Agent.

    Uses a dictionary to store Q-values for state-action pairs.
    State can be any hashable type (typically a tuple).
    """

    def __init__(
        self,
        n_actions: int = 7,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        """
        Initialize the Q-learning agent.

        Args:
            n_actions: Number of possible actions (default 7 for MiniGrid).
            learning_rate: Alpha - how quickly to update Q-values.
            discount_factor: Gamma - importance of future rewards.
            epsilon: Initial exploration rate for epsilon-greedy policy.
            epsilon_decay: Factor to decay epsilon after each episode.
            epsilon_min: Minimum value for epsilon.
        """
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: dictionary mapping (state, action) -> Q-value
        # Using defaultdict to initialize unseen states to 0
        self.q_table: Dict[Tuple[Any, int], float] = defaultdict(float)

        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_q_changes = []

    def get_action(self, state: Tuple, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.

        Args:
            state: Current state (must be hashable).
            training: If True, use epsilon-greedy. If False, use greedy policy.

        Returns:
            Selected action index.
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: action with highest Q-value
            return self._get_best_action(state)

    def _get_best_action(self, state: Tuple) -> int:
        """Get the action with highest Q-value for a given state."""
        q_values = [self.q_table[(state, a)] for a in range(self.n_actions)]
        return int(np.argmax(q_values))

    def update(
        self,
        state: Tuple,
        action: int,
        reward: float,
        next_state: Tuple,
        terminated: bool,
    ) -> float:
        """
        Update Q-value using the Q-learning update rule.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state after taking action.
            terminated: Whether the episode ended.

        Returns:
            The Q-value change (for monitoring convergence).
        """
        current_q = self.q_table[(state, action)]

        # Calculate target Q-value
        if terminated:
            # No future rewards if episode ended
            target_q = reward
        else:
            # Max Q-value for next state
            next_q_values = [self.q_table[(next_state, a)] for a in range(self.n_actions)]
            max_next_q = max(next_q_values)
            target_q = reward + self.discount_factor * max_next_q

        # Q-learning update
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[(state, action)] = new_q

        return abs(new_q - current_q)

    def decay_epsilon(self):
        """Decay the exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_q_table_size(self) -> int:
        """Return the number of state-action pairs in Q-table."""
        return len(self.q_table)

    def get_q_values_for_state(self, state: Tuple) -> np.ndarray:
        """Get all Q-values for a given state."""
        return np.array([self.q_table[(state, a)] for a in range(self.n_actions)])

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
        print(f"  States in Q-table: {self.get_q_table_size() // self.n_actions}")
        print(f"  Current epsilon: {self.epsilon:.4f}")

    def reset_stats(self):
        """Reset training statistics."""
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_q_changes = []
