"""
CustomAgent13 - Tabular Q-Learning with Visited-State Penalty (Intrinsic Motivation)
====================================================================================

This agent adds a loop detection mechanism:
- Tracks visited (x, y) positions within each episode
- Penalizes revisiting the same position (-0.1 by default)
- Forces the agent to escape loops and explore new areas

Combined with curriculum learning and tie-breaking action selection,
this significantly reduces spinning behavior.

State representation: Same as CustomAgent11/12
(dir, dxg, dyg, dxw, dyw, front_blocked, left_blocked, right_blocked)
"""

import random
import pickle
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, Any, List


class CustomAgent13:
    """
    Tabular Q-Learning with visited-state penalty for loop avoidance.
    """

    def __init__(
        self,
        n_actions: int = 3,
        learning_rate: float = 0.3,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.9995,
        epsilon_min: float = 0.01,
        epsilon_delay: int = 1350,
        distance_reward_scale: float = 0.0005,
        coord_clamp: int = 4,
        wall_clamp: int = 1,
        loop_penalty: float = -0.1,  # NEW: Penalty for revisiting positions
    ):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_delay_episodes = epsilon_delay
        self.distance_reward_scale = distance_reward_scale
        self.coord_clamp = coord_clamp
        self.wall_clamp = wall_clamp
        self.loop_penalty = loop_penalty
        self.episode_count = 0

        self.q_table: Dict[Tuple[Any, int], float] = defaultdict(float)
        self.episode_rewards = []
        self.episode_lengths = []
        
        # NEW: Track visited positions within episode
        self.visited_positions: set = set()

    def _clamp(self, value: int, limit: int) -> int:
        """Clamp value to [-limit, +limit]."""
        return max(-limit, min(limit, value))

    # Direction offsets: 0=right, 1=down, 2=left, 3=up
    DIR_DELTAS = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}

    def _get_cell_in_dir(self, agent_pos: Tuple[int, int], direction: int) -> Tuple[int, int]:
        ax, ay = int(agent_pos[0]), int(agent_pos[1])
        dx, dy = self.DIR_DELTAS[direction % 4]
        return (ax + dx, ay + dy)

    def _is_blocked(self, cell: Tuple[int, int], obstacle_positions) -> int:
        cx, cy = int(cell[0]), int(cell[1])
        if cx < 1 or cx > 10 or cy < 1 or cy > 10:
            return 1
        if obstacle_positions:
            for ox, oy in obstacle_positions:
                if int(ox) == cx and int(oy) == cy:
                    return 1
        return 0

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
        Build hybrid state: precise goal + coarse wall + blocked sensors.
        
        State: (dir, dxg, dyg, dxw, dyw, front, left, right)
        """
        ax, ay = int(agent_pos[0]), int(agent_pos[1])
        gx, gy = int(goal_pos[0]), int(goal_pos[1])

        # Precise goal vector (clamp=4)
        dxg = self._clamp(gx - ax, self.coord_clamp)
        dyg = self._clamp(gy - ay, self.coord_clamp)

        # Coarse wall direction (clamp=1: just -1/0/+1)
        dxw, dyw = 0, 0
        if obstacle_positions:
            ox, oy = min(
                obstacle_positions,
                key=lambda p: (int(p[0]) - ax) ** 2 + (int(p[1]) - ay) ** 2
            )
            dxw = self._clamp(int(ox) - ax, self.wall_clamp)
            dyw = self._clamp(int(oy) - ay, self.wall_clamp)

        # Tri-directional blocked sensors
        front_blocked = self._is_blocked(self._get_cell_in_dir(agent_pos, agent_dir), obstacle_positions)
        left_blocked = self._is_blocked(self._get_cell_in_dir(agent_pos, (agent_dir - 1) % 4), obstacle_positions)
        right_blocked = self._is_blocked(self._get_cell_in_dir(agent_pos, (agent_dir + 1) % 4), obstacle_positions)

        return (int(agent_dir), dxg, dyg, dxw, dyw, front_blocked, left_blocked, right_blocked)

    def get_action(self, state: Tuple, training: bool = True, front_blocked: int = 0) -> int:
        """
        Get action with optional action masking.
        
        Args:
            state: The current state tuple
            training: Whether to use epsilon-greedy exploration
            front_blocked: If 1, mask out forward action (can't move into wall)
        """
        # Action IDs: 0=turn_left, 1=turn_right, 2=move_forward
        FORWARD = 2
        
        # Build list of valid actions (mask out forward if blocked)
        valid_actions = [0, 1, 2]  # All actions by default
        if front_blocked == 1:
            valid_actions = [0, 1]  # Only turn left/right
        
        if training and np.random.random() < self.epsilon:
            # Epsilon-greedy exploration with action masking
            return int(np.random.choice(valid_actions))
        
        return self._get_best_action(state, valid_actions)

    def _get_best_action(self, state: Tuple, valid_actions: List[int] = None) -> int:
        """
        Get best action with tie-breaking preference and action masking.
        
        When Q-values are equal (common in corners/obstacles), prefer:
        1. FORWARD (2) - try to move
        2. LEFT (0)    - turn left
        3. RIGHT (1)   - turn right
        """
        # Action IDs: 0=turn_left, 1=turn_right, 2=move_forward
        FORWARD, LEFT, RIGHT = 2, 0, 1
        PREFERRED_ORDER = [FORWARD, LEFT, RIGHT]
        
        if valid_actions is None:
            valid_actions = [0, 1, 2]
        
        # Get Q-values only for valid actions
        action_q_pairs = [(a, self.q_table[(state, a)]) for a in valid_actions]
        max_q = max(q for a, q in action_q_pairs)
        best_actions = [a for a, q in action_q_pairs if q == max_q]
        
        # If only one best action, use it
        if len(best_actions) == 1:
            return best_actions[0]
        
        # Tie-breaking: prefer forward > left > right (among valid actions)
        for action in PREFERRED_ORDER:
            if action in best_actions:
                return action
        
        # Fallback (shouldn't reach here)
        return int(np.random.choice(best_actions))

    def update(self, state, action, reward, next_state, terminated) -> float:
        current_q = self.q_table[(state, action)]
        if terminated:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * max(
                self.q_table[(next_state, a)] for a in range(self.n_actions)
            )
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[(state, action)] = new_q
        return abs(new_q - current_q)

    def decay_epsilon(self):
        self.episode_count += 1
        if self.episode_count > self.epsilon_delay_episodes:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # NEW: Visited-state penalty methods
    def check_visited_penalty(self, agent_pos: Tuple[int, int]) -> float:
        """
        Check if position was visited before and return penalty.
        Call this after each step in the training loop.
        """
        pos = (int(agent_pos[0]), int(agent_pos[1]))
        if pos in self.visited_positions:
            return self.loop_penalty
        else:
            self.visited_positions.add(pos)
            return 0.0

    def reset_visited(self):
        """Reset visited positions at episode end."""
        self.visited_positions.clear()

    def get_q_table_size(self) -> int:
        return len(self.q_table)

    def get_state_count(self) -> int:
        return len(self.q_table) // self.n_actions

    def save(self, filepath: str):
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
                "wall_clamp": self.wall_clamp,
                "loop_penalty": self.loop_penalty,
            },
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"Q-table saved to {filepath}")

    def load(self, filepath: str):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(float, data["q_table"])
        self.epsilon = data["epsilon"]
        self.episode_rewards = data.get("episode_rewards", [])
        self.episode_lengths = data.get("episode_lengths", [])
        hyperparams = data.get("hyperparameters", {})
        self.coord_clamp = hyperparams.get("coord_clamp", 4)
        self.wall_clamp = hyperparams.get("wall_clamp", 1)
        self.loop_penalty = hyperparams.get("loop_penalty", -0.1)
        print(f"Q-table loaded from {filepath}")

    def reset_stats(self):
        self.episode_rewards = []
        self.episode_lengths = []
