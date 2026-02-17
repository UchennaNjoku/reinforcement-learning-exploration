"""
CustomAgent11 - HYBRID: Wall Vector + Tri-Directional Sensors
=============================================================

LESSONS LEARNED:
- v9 (dxw,dyw + front_blocked only): 36% — knew where wall was but not if sides blocked
- v10 (front/left/right blocked only): 23% — knew local blockage but lost wall position info

The problem is clear:
- dxw/dyw tells agent WHERE the wall is (needed for "go around" strategy)
- left/right_blocked tells agent WHICH TURNS ARE SAFE (needed to avoid wall hits)
- You need BOTH.

SOLUTION: Keep both, but reduce clamp to manage state space
-----------------------------------------------------------

State: (dir, dxg, dyg, dxw, dyw, front_blocked, left_blocked, right_blocked)

With coord_clamp=3 (not 4):
- dir: 4
- dxg, dyg: 7 each (-3 to +3)  
- dxw, dyw: 7 each (-3 to +3)
- front/left/right_blocked: 2 each

Total: 4 × 7 × 7 × 7 × 7 × 2 × 2 × 2 = 76,832 states

That's larger, but with 15-20k episodes the agent should visit
the states that matter (states near walls + near goals).

Alternative with clamp=2:
4 × 5 × 5 × 5 × 5 × 2 × 2 × 2 = 20,000 states — very learnable!

We'll use clamp=2 for the wall vector and clamp=4 for goal vector.
This makes sense: you need fine goal direction but coarse wall direction.

State: (dir, dxg, dyg, dxw_coarse, dyw_coarse, front, left, right)
- dir: 4
- dxg, dyg: 9 each (clamp=4)
- dxw, dyw: 5 each (clamp=2)  
- blocked bits: 2 each

Total: 4 × 9 × 9 × 5 × 5 × 2 × 2 × 2 = 64,800 states

Hmm still large. Let's try clamp=3 for goal, clamp=2 for wall:
4 × 7 × 7 × 5 × 5 × 2 × 2 × 2 = 39,200 states

Or simplest: just add left/right blocked to v9 with clamp=3:
4 × 7 × 7 × 7 × 7 × 2 × 2 × 2 = 76,832

Let's go with a SMART compromise:
- Goal vector: clamp=4 (9 values) — need precision for navigation
- Wall vector: clamp=1 (3 values: -1, 0, +1) — just need direction
- 3 blocked bits

4 × 9 × 9 × 3 × 3 × 2 × 2 × 2 = 23,328 states

This is the sweet spot! Agent knows:
- Precisely where goal is (9×9 = 81 relative positions)
- Roughly where nearest wall is (left/right/same, above/below/same)
- Which immediate directions are blocked (8 combinations)

~23k states is very learnable in 12-15k episodes.
"""

import random
import pickle
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, Any, List


class CustomAgent11:
    """
    Tabular Q-Learning with wall vector + tri-directional sensors.
    
    State: (dir, dxg, dyg, dxw_coarse, dyw_coarse, front, left, right)
    ~23k states — combines position awareness with local obstacle detection.
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
        coord_clamp: int = 4,            # Used for goal vector
        wall_clamp: int = 1,             # Coarse wall direction (-1, 0, +1)
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
        self.episode_count = 0

        self.q_table: Dict[Tuple[Any, int], float] = defaultdict(float)
        self.episode_rewards = []
        self.episode_lengths = []

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
        - dir: 4 values
        - dxg, dyg: ±4 (9 values each) — precise goal direction
        - dxw, dyw: ±1 (3 values each) — coarse wall direction  
        - front/left/right: 0/1 — immediate blockage
        
        Total: 4 × 9 × 9 × 3 × 3 × 2 × 2 × 2 = 23,328 states
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

    def get_action(self, state: Tuple, training: bool = True) -> int:
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return self._get_best_action(state)

    def _get_best_action(self, state: Tuple) -> int:
        q_values = [self.q_table[(state, a)] for a in range(self.n_actions)]
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
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
        print(f"Q-table loaded from {filepath}")

    def reset_stats(self):
        self.episode_rewards = []
        self.episode_lengths = []