"""
Tabular Q-Learning Agent with Continuous Distance Features
==========================================================

This agent uses Q-learning with a state representation that includes:
- Agent position (x, y) 
- Goal position (x, y)
- Euclidean distance to goal (discretized to 1 decimal for manageable state space)
- Euclidean distance to nearest obstacle (discretized to 0.5 for precision)

Key Insight: The agent learns that being CLOSE to walls is okay (for navigating 
around them), but COLLIDING with walls is bad. The distance feature helps it 
learn optimal paths that may involve riding along walls.
"""

from __future__ import annotations

import numpy as np
import pickle
from collections import defaultdict
from typing import Dict, Tuple, Any


class CustomAgent:
    """
    Tabular Q-Learning Agent with Fine-Grained Distance Features.
    
    State Representation:
        (agent_x, agent_y, goal_x, goal_y, dist_to_goal_rounded, dist_to_obstacle_rounded)
        
    Distance Precision:
        - Goal distance: rounded to nearest 1.0 (0, 1, 2, 3, ... 14)
        - Obstacle distance: rounded to nearest 0.5 (0, 0.5, 1.0, 1.5, 2.0, ...)
        
    This preserves enough precision for the agent to learn:
    - When it's getting closer to goal (progress signal)
    - When it's right next to a wall (0 or 0.5) vs safely away (2+)
    - That riding along walls (distance ~0.5-1.0) can be optimal
    
    Action Space: 3 actions (0=turn_left, 1=turn_right, 2=move_forward)
    """

    def __init__(
        self,
        n_actions: int = 3,
        learning_rate: float = 0.1,      # Alpha: how fast to learn
        discount_factor: float = 0.99,   # Gamma: importance of future rewards
        epsilon: float = 1.0,            # Initial exploration rate
        epsilon_decay: float = 0.995,    # Decay per episode
        epsilon_min: float = 0.01,       # Minimum exploration
    ):
        """Initialize the Q-learning agent."""
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # ============================================================
        # Q-TABLE DATA STRUCTURE
        # ============================================================
        # Maps (state_tuple, action) -> Q-value
        # State: (ax, ay, gx, gy, dist_goal_rounded, dist_obs_rounded)
        self.q_table: Dict[Tuple[Any, int], float] = defaultdict(float)
        
        # ============================================================
        # DISTANCE PRECISION SETTINGS
        # ============================================================
        # How much to round distances for state discretization
        # Smaller = more precise but larger state space
        self.goal_dist_precision = 1.0      # Round to nearest integer
        self.obstacle_dist_precision = 0.5  # Round to nearest 0.5 (finer for obstacles)
        
        # ============================================================
        # TRAINING STATISTICS
        # ============================================================
        self.episode_rewards = []
        self.episode_lengths = []

    # ========================================================================
    # STATE REPRESENTATION: Build state with precise distance features
    # ========================================================================
    
    def _round_distance(self, distance: float, precision: float) -> float:
        """
        Round distance to specified precision for state discretization.
        
        Args:
            distance: Continuous distance value
            precision: Rounding precision (e.g., 1.0 for integer, 0.5 for half-steps)
            
        Returns:
            Rounded distance value
        """
        return round(distance / precision) * precision
    
    def build_state(
        self,
        agent_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        dist_to_goal: float,
        dist_to_obstacle: float
    ) -> Tuple:
        """
        Build the state tuple from environment observations.
        
        Uses fine-grained distance rounding to preserve information about:
        - Exact-ish distance to goal (for progress tracking)
        - Proximity to walls (for learning wall-riding is okay)
        
        Args:
            agent_pos: (x, y) agent position
            goal_pos: (x, y) goal position
            dist_to_goal: Euclidean distance to goal (continuous)
            dist_to_obstacle: Min distance to any wall obstacle (continuous)
            
        Returns:
            State tuple: (ax, ay, gx, gy, dist_goal_rounded, dist_obs_rounded)
        """
        # Round distances for manageable state space
        # But keep enough precision for the agent to learn wall-riding
        goal_dist_rounded = self._round_distance(dist_to_goal, self.goal_dist_precision)
        obs_dist_rounded = self._round_distance(dist_to_obstacle, self.obstacle_dist_precision)
        
        # Build state tuple
        state = (
            int(agent_pos[0]),          # Agent X
            int(agent_pos[1]),          # Agent Y
            int(goal_pos[0]),           # Goal X
            int(goal_pos[1]),           # Goal Y
            goal_dist_rounded,          # Distance to goal (rounded to 1.0)
            obs_dist_rounded,           # Distance to obstacle (rounded to 0.5)
        )
        
        return state

    # ========================================================================
    # ACTION SELECTION: Epsilon-Greedy Policy
    # ========================================================================
    
    def get_action(self, state: Tuple, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        With probability epsilon: explore (random action)
        With probability 1-epsilon: exploit (best known action)
        """
        if training and np.random.random() < self.epsilon:
            # EXPLORE: Random action for discovery
            return np.random.randint(self.n_actions)
        else:
            # EXPLOIT: Best known action for this state
            return self._get_best_action(state)

    def _get_best_action(self, state: Tuple) -> int:
        """
        Get the action with highest Q-value for given state.
        Breaks ties randomly for better exploration.
        """
        # Get Q-values for all actions in this state
        q_values = [self.q_table[(state, a)] for a in range(self.n_actions)]
        
        # Find max Q-value
        max_q = max(q_values)
        
        # Find all actions with max Q (for tie-breaking)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        
        # Random tie-breaker
        return int(np.random.choice(best_actions))

    # ========================================================================
    # LEARNING UPDATE: Q-Learning Algorithm
    # ========================================================================
    
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
        
        Q-Learning Update:
            Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
            
        The distance features help the agent learn:
        - Decreasing goal distance → higher Q-values (good progress)
        - Small obstacle distance isn't always bad (wall-riding can be optimal)
        - Only obstacle collision (distance ~0 with negative reward) is truly bad
        
        Args:
            state: Current state (s)
            action: Action taken (a)
            reward: Reward received (r)
            next_state: Next state after action (s')
            terminated: True if episode ended (reached goal or failed)
            
        Returns:
            Absolute change in Q-value (for monitoring convergence)
        """
        # Current Q-value for (state, action)
        current_q = self.q_table[(state, action)]
        
        # Calculate target Q-value
        if terminated:
            # Terminal state: no future rewards
            target_q = reward
        else:
            # Non-terminal: include discounted future rewards
            next_q_values = [self.q_table[(next_state, a)] for a in range(self.n_actions)]
            max_next_q = max(next_q_values)
            target_q = reward + self.discount_factor * max_next_q
        
        # Q-Learning update
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[(state, action)] = new_q
        
        return abs(new_q - current_q)

    # ========================================================================
    # EXPLORATION DECAY
    # ========================================================================
    
    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ========================================================================
    # PERSISTENCE: Save and Load Q-Table
    # ========================================================================
    
    def save(self, filepath: str):
        """Save the Q-table and hyperparameters to file."""
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
                "goal_dist_precision": self.goal_dist_precision,
                "obstacle_dist_precision": self.obstacle_dist_precision,
            },
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"Q-table saved to {filepath}")
        print(f"  States in Q-table: {len(self.q_table) // self.n_actions}")

    def load(self, filepath: str):
        """Load the Q-table and hyperparameters from file."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        self.q_table = defaultdict(float, data["q_table"])
        self.epsilon = data["epsilon"]
        self.episode_rewards = data.get("episode_rewards", [])
        self.episode_lengths = data.get("episode_lengths", [])
        
        # Restore precision settings if saved
        hyperparams = data.get("hyperparameters", {})
        self.goal_dist_precision = hyperparams.get("goal_dist_precision", 1.0)
        self.obstacle_dist_precision = hyperparams.get("obstacle_dist_precision", 0.5)
        
        print(f"Q-table loaded from {filepath}")
        print(f"  States in Q-table: {len(self.q_table) // self.n_actions}")
        print(f"  Current epsilon: {self.epsilon:.4f}")

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_q_table_size(self) -> int:
        """Return number of state-action pairs in Q-table."""
        return len(self.q_table)
    
    def get_state_count(self) -> int:
        """Return number of unique states visited."""
        return len(self.q_table) // self.n_actions
    
    def get_q_values_for_state(self, state: Tuple) -> np.ndarray:
        """Get Q-values for all actions in a given state."""
        return np.array([self.q_table[(state, a)] for a in range(self.n_actions)])
    
    def reset_stats(self):
        """Reset training statistics."""
        self.episode_rewards = []
        self.episode_lengths = []


# =============================================================================
# ALGORITHM EXPLANATION: Why This Works for Wall-Riding
# =============================================================================
"""
DISTANCE-AWARE Q-LEARNING WITH FINE PRECISION
=============================================

WHY FINE-GRAINED DISTANCES?
---------------------------
Coarse bins (0=danger, 1=safe) teach the agent to "always avoid walls"
but that's suboptimal! Sometimes the shortest path requires:
- Going around a wall (riding along it)
- Passing through narrow gaps between walls and borders
- Being temporarily close to obstacles to reach the goal faster

Our fine-grained approach (rounded to 0.5 for obstacles):
- 0.0 = touching/colliding with wall (bad, negative reward)
- 0.5 = right next to wall (can be good for navigating around)
- 1.0 = one tile away from wall (comfortable for wall-riding)
- 1.5+ = safely away from walls

The agent learns through experience that:
- Distance 0.5 + moving parallel to wall = OK (optimal path)
- Distance 0.5 + moving toward wall = bad (will hit wall soon)
- Distance 2.0 + moving toward goal = good (safe and progressing)

STATE SPACE SIZE
----------------
With our precision settings:
- Grid positions: 10x10 = 100 each for agent and goal
- Goal distance: 0-14 (rounded to 1.0) = ~15 values
- Obstacle distance: 0, 0.5, 1.0, 1.5, 2.0, ... = ~10 values

This is manageable because the agent only visits a fraction of possible states.

WALL-RIDING BEHAVIOR
--------------------
The agent learns wall-riding through Q-value updates:

Scenario: Wall blocking direct path to goal

State A: Agent next to wall (obs_dist=0.5), facing along wall
  Action "forward" → stays at obs_dist=0.5, progresses toward goal
  Q-value increases because: reward + future_goal_proximity is good

State B: Agent next to wall (obs_dist=0.5), facing toward wall  
  Action "forward" → hits wall (obs_dist=0.0), negative reward
  Q-value decreases because: collision is bad

Result: Agent learns to ride along walls but not crash into them!

EUCLIDEAN VS MANHATTAN DISTANCE
-------------------------------
We use Euclidean distance (straight-line) because:
1. More natural for diagonal movement consideration
2. Standard in most RL environments
3. Gives smoother gradient as agent moves

However, since the agent can only move in cardinal directions 
(up/down/left/right), Manhattan distance would also work.
"""
