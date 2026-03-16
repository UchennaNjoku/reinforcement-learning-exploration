"""Research-facing wrapper around PettingZoo pursuit_v4."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np


if os.environ.get("PYGAME_HIDE_SUPPORT_PROMPT") is None:
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from pettingzoo.sisl import pursuit_v4


GridPos = tuple[int, int]


@dataclass(frozen=True)
class MapSpec:
    name: str
    grid: np.ndarray
    pursuer_starts: tuple[GridPos, GridPos, GridPos]
    evader_starts: tuple[GridPos, ...]


def _empty_map(size: int) -> np.ndarray:
    return np.zeros((size, size), dtype=np.int32)


def _with_vertical_barrier(
    size: int,
    x: int,
    blocked_rows: range,
    gap_rows: tuple[int, ...],
) -> np.ndarray:
    grid = _empty_map(size)
    for y in blocked_rows:
        if y not in gap_rows:
            grid[x, y] = -1
    return grid


def _with_central_block(size: int, start: int, end: int) -> np.ndarray:
    grid = _empty_map(size)
    grid[start:end, start:end] = -1
    return grid


MAP_SPECS: dict[str, MapSpec] = {
    "easy_open": MapSpec(
        name="easy_open",
        grid=_empty_map(16),
        pursuer_starts=((2, 2), (2, 13), (5, 8)),
        evader_starts=((12, 8),),
    ),
    "center_block": MapSpec(
        name="center_block",
        grid=_with_central_block(size=16, start=6, end=10),
        pursuer_starts=((2, 3), (2, 12), (5, 8)),
        evader_starts=((12, 8),),
    ),
    "split_barrier": MapSpec(
        name="split_barrier",
        grid=_with_vertical_barrier(
            size=16,
            x=8,
            blocked_rows=range(2, 14),
            gap_rows=(5, 10),
        ),
        pursuer_starts=((2, 4), (2, 11), (4, 8)),
        evader_starts=((13, 8),),
    ),
}


class FixedMapPursuitWrapper:
    """Parallel-style wrapper with fixed maps, fixed starts, and cleaner rewards."""

    def __init__(
        self,
        map_name: str = "center_block",
        render_mode: str | None = None,
        max_cycles: int = 300,
        obs_range: int = 7,
        step_penalty: float = -0.01,
        catch_reward: float = 1.0,
        tag_reward: float = 0.0,
        n_catch: int = 1,
        surround: bool = True,
        shared_reward: bool = True,
        freeze_evaders: bool = False,
        distance_reward_scale: float = 0.0,
    ) -> None:
        if map_name not in MAP_SPECS:
            available = ", ".join(sorted(MAP_SPECS))
            raise ValueError(f"Unknown map_name={map_name!r}. Available: {available}")

        self.map_spec = MAP_SPECS[map_name]
        self._grid_size = self.map_spec.grid.shape[0]
        self._distance_reward_scale = distance_reward_scale
        self.env = pursuit_v4.parallel_env(
            x_size=self.map_spec.grid.shape[0],
            y_size=self.map_spec.grid.shape[1],
            max_cycles=max_cycles,
            shared_reward=shared_reward,
            n_evaders=len(self.map_spec.evader_starts),
            n_pursuers=len(self.map_spec.pursuer_starts),
            obs_range=obs_range,
            n_catch=n_catch,
            freeze_evaders=freeze_evaders,
            tag_reward=tag_reward,
            catch_reward=catch_reward,
            urgency_reward=step_penalty,
            surround=surround,
            render_mode=render_mode,
        )
        self.possible_agents = list(self.env.possible_agents)
        self.agents: list[str] = []
        self.last_positions: dict[str, GridPos] = {}

    def action_space(self, agent: str):
        return self.env.action_space(agent)

    def observation_space(self, agent: str):
        return self.env.observation_space(agent)

    def render(self):
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def _raw_env(self):
        return self.env.unwrapped

    def _base_env(self):
        return self._raw_env().env

    def _validate_positions(self) -> None:
        grid = self.map_spec.grid
        for label, positions in (
            ("pursuer", self.map_spec.pursuer_starts),
            ("evader", self.map_spec.evader_starts),
        ):
            for pos in positions:
                x, y = pos
                if not (0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]):
                    raise ValueError(f"{label} start {pos} is out of bounds")
                if grid[x, y] == -1:
                    raise ValueError(f"{label} start {pos} is inside an obstacle")

    def _apply_fixed_layout(self) -> None:
        self._validate_positions()
        base = self._base_env()
        raw = self._raw_env()

        fixed_grid = self.map_spec.grid.copy()
        base.map_matrix = fixed_grid
        for agent in base.pursuers:
            agent.map_matrix = fixed_grid
        for agent in base.evaders:
            agent.map_matrix = fixed_grid

        for idx, (x, y) in enumerate(self.map_spec.pursuer_starts):
            base.pursuer_layer.set_position(idx, x, y)
        for idx, (x, y) in enumerate(self.map_spec.evader_starts):
            base.evader_layer.set_position(idx, x, y)

        base.model_state[0] = fixed_grid
        base.model_state[1] = base.pursuer_layer.get_state_matrix()
        base.model_state[2] = base.evader_layer.get_state_matrix()
        base.latest_reward_state = np.zeros(base.num_agents, dtype=np.float32)
        base.latest_done_state = [False for _ in range(base.num_agents)]
        raw.rewards = {agent: 0.0 for agent in raw.agents}
        raw._cumulative_rewards = {agent: 0.0 for agent in raw.agents}

    def _collect_positions(self) -> dict[str, GridPos]:
        base = self._base_env()
        return {
            agent: tuple(int(v) for v in base.pursuer_layer.get_position(idx))
            for idx, agent in enumerate(self.possible_agents)
        }

    def _evader_positions(self) -> list[GridPos]:
        base = self._base_env()
        return [
            tuple(int(v) for v in base.evader_layer.get_position(i))
            for i in range(base.evader_layer.n_agents())
        ]

    def _distance_reward(self, pursuer_pos: GridPos) -> float:
        """Manhattan distance penalty toward nearest evader, normalized to [-1, 0]."""
        evaders = self._evader_positions()
        if not evaders:
            return 0.0
        px, py = pursuer_pos
        min_dist = min(abs(px - ex) + abs(py - ey) for ex, ey in evaders)
        max_dist = 2 * (self._grid_size - 1)
        return -self._distance_reward_scale * (min_dist / max_dist)

    def _observations(self) -> dict[str, np.ndarray]:
        return {
            agent: self.env.aec_env.observe(agent)
            for agent in self.env.agents
        }

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        observations, infos = self.env.reset(seed=seed, options=options)
        del observations

        self._apply_fixed_layout()
        self.agents = list(self.env.agents)
        self.last_positions = self._collect_positions()
        fixed_infos = {
            agent: {
                **infos.get(agent, {}),
                "map_name": self.map_spec.name,
                "start_position": list(self.last_positions[agent]),
            }
            for agent in self.env.agents
        }
        return self._observations(), fixed_infos

    def step(
        self,
        actions: dict[str, int],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        prev_positions = self._collect_positions()
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        self.agents = list(self.env.agents)
        new_positions = self._collect_positions()
        base = self._base_env()
        evaders_remaining = int(base.evader_layer.n_agents())

        wrapped_infos: dict[str, dict[str, Any]] = {}
        for agent in rewards:
            attempted_action = int(actions[agent])
            moved = prev_positions[agent] != new_positions[agent]
            blocked = attempted_action != 4 and not moved
            wrapped_infos[agent] = {
                **infos.get(agent, {}),
                "map_name": self.map_spec.name,
                "position": list(new_positions[agent]),
                "attempted_action": attempted_action,
                "blocked_move": blocked,
                "evaders_remaining": evaders_remaining,
            }

        self.last_positions = new_positions
        clean_rewards = {
            agent: float(reward) + (
                self._distance_reward(new_positions[agent])
                if self._distance_reward_scale > 0.0
                else 0.0
            )
            for agent, reward in rewards.items()
        }
        return observations, clean_rewards, terminations, truncations, wrapped_infos


def make_fixed_pursuit_env(
    map_name: str = "center_block",
    render_mode: str | None = None,
    **kwargs: Any,
) -> FixedMapPursuitWrapper:
    return FixedMapPursuitWrapper(map_name=map_name, render_mode=render_mode, **kwargs)

