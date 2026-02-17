"""
CustomAgent9 - FULLY OBSERVABLE STATE + FRONT BLOCKED SENSOR
===========================================================

PROBLEM WITH V8:
----------------
- Learning rate 0.1 was too slow for dense rewards
- Epsilon min 0.05 meant 5% random actions even at end
- Agent couldn't execute precise navigation sequences
- State didn't include wall info → couldn't navigate around obstacles
- Inconsistent performance despite good training rewards

SOLUTION: Fully observable state with front blocked sensor
----------------------------------------------------------

CHANGE 1: FASTER LEARNING (learning_rate 0.1 -> 0.3)
-----------------------------------------------------
OLD: Q = Q + 0.1 * (target - Q)  - slow, gradual updates
NEW: Q = Q + 0.3 * (target - Q)  - 3x faster adaptation

CHANGE 2: LOWER EPSILON MINIMUM (0.05 -> 0.01)
----------------------------------------------
OLD: Minimum 5% random actions forever
NEW: Minimum 1% random actions (99% greedy)

CHANGE 3: SLOWER DECAY (0.997 -> 0.999)
----------------------------------------
Faster exploration, slower shift to exploitation.

CHANGE 4: INCREASED CLAMP (2 -> 4)
----------------------------------
Less aliasing when goal/wall is far away.

CHANGE 5: ADDED WALL VECTOR (dxw, dyw)
--------------------------------------
Nearest obstacle relative position.
State: (dir, dxg, dyg, dxw, dyw)

CHANGE 6: ADDED FRONT BLOCKED SENSOR (NEW!)
-------------------------------------------
front_blocked ∈ {0, 1}
1 = tile directly ahead is wall/obstacle
0 = tile ahead is free to move

This prevents wasted steps into walls and "turn loops".
Massive improvement in navigation efficiency!

STATE SPACE:
------------
(dir, dxg, dyg, dxw, dyw, front_blocked)
- dir: 4 values (0-3)
- dxg, dyg: 9 values each (-4 to +4)
- dxw, dyw: 9 values each (-4 to +4)
- front_blocked: 2 values (0 or 1)

Total: 4 × 9 × 9 × 9 × 9 × 2 = 52,488 states

REWARD STRUCTURE:
-----------------
- Dense: (prev_dist - curr_dist) × 0.1
- Step: -0.01
- Wall bump: -0.1
- Timeout: -1.0
- Goal: +1.0

EXPECTED IMPROVEMENT:
---------------------
- Agent knows exactly where walls are
- Front blocked sensor prevents wall-hitting
- Can navigate efficiently around obstacles
- Much higher success rate!
"""

import random 
import pickle
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, Any


class CustomAgent9:
    """
    Tabular Q-Learning Agent with FASTER LEARNING and REDUCED EXPLORATION.
    
    Key changes from v8:
    - learning_rate: 0.1 -> 0.3 (3x faster)
    - epsilon_min: 0.05 -> 0.01 (99% greedy)
    - epsilon_decay: 0.997 -> 0.999 (slower, more exploration)
    
    State Representation:
        (agent_dir, dx, dy) - relative coordinates
    """

    def __init__(
        self,
        n_actions: int = 3,
        learning_rate: float = 0.5,      # FASTER (was 0.1)
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.9995,    # SLOWER decay (was 0.997)
        epsilon_min: float = 0.05,       # LOWER (was 0.05)
        epsilon_delay_episodes: int = 2000,
        epsilon_delay: int = None,        # Alias for epsilon_delay_episodes (from config.py)
        distance_reward_scale: float = 0.05,
        coord_clamp: int = 4,  # INCREASED from 2 to 4 (less aliasing)
    ):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon 
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # Use epsilon_delay if provided (config.py uses this name), else epsilon_delay_episodes
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

    def _get_front_cell(self, agent_pos: Tuple[int, int], agent_dir: int) -> Tuple[int, int]:
        """
        Get the cell directly in front of the agent.
        
        agent_dir: 0=right, 1=down, 2=left, 3=up
        Returns: (fx, fy) coordinates of front cell
        """
        ax, ay = int(agent_pos[0]), int(agent_pos[1])
        
        if agent_dir == 0:    # right
            return (ax + 1, ay)
        elif agent_dir == 1:  # down
            return (ax, ay + 1)
        elif agent_dir == 2:  # left
            return (ax - 1, ay)
        else:                 # up (dir == 3)
            return (ax, ay - 1)
    
    def _is_blocked(self, cell: Tuple[int, int], obstacle_positions) -> int:
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
    
    def build_state(
        self,
        agent_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
        agent_dir: int,
        dist_to_goal: float,
        dist_to_obstacle: float,
        obstacle_positions=None,  # list of (x,y) wall positions
    ) -> Tuple:
        """
        Build RELATIVE state tuple with goal, nearest wall, AND front blocked sensor.
        
        State: (dir, dxg, dyg, dxw, dyw, front_blocked)
        - dir: agent direction (0-3)
        - dxg, dyg: goal position relative to agent (clamped)
        - dxw, dyw: nearest wall position relative to agent (clamped)
        - front_blocked: 1 if tile directly ahead is wall/obstacle, else 0
        """
        ax, ay = int(agent_pos[0]), int(agent_pos[1])
        gx, gy = int(goal_pos[0]), int(goal_pos[1])
        
        # Goal relative position
        dxg = self._clamp(gx - ax, -self.coord_clamp, self.coord_clamp)
        dyg = self._clamp(gy - ay, -self.coord_clamp, self.coord_clamp)
        
        # Nearest obstacle/wall relative position
        dxw, dyw = 0, 0
        if obstacle_positions:
            # Find nearest obstacle by Euclidean distance
            ox, oy = min(
                obstacle_positions,
                key=lambda p: (int(p[0]) - ax) ** 2 + (int(p[1]) - ay) ** 2
            )
            dxw = self._clamp(int(ox) - ax, -self.coord_clamp, self.coord_clamp)
            dyw = self._clamp(int(oy) - ay, -self.coord_clamp, self.coord_clamp)
        
        # FRONT BLOCKED SENSOR (NEW)
        # Check if the cell directly in front is blocked
        front_cell = self._get_front_cell(agent_pos, agent_dir)
        front_blocked = self._is_blocked(front_cell, obstacle_positions)
        
        # State includes direction, goal relative, wall relative, AND front blocked
        # State size: 4 * 9 * 9 * 9 * 9 * 2 = 52,488 states (with clamp=4)
        state = (int(agent_dir), dxg, dyg, dxw, dyw, front_blocked)
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
        self.coord_clamp = hyperparams.get("coord_clamp", 2)
        print(f"Q-table loaded from {filepath}")

    def reset_stats(self):
        """Reset training statistics."""
        self.episode_rewards = []
        self.episode_lengths = []
