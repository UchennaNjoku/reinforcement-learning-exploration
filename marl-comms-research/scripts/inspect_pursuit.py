"""Inspect raw observations and action spaces for PettingZoo pursuit_v4."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


# pursuit_v4 initializes pygame during env construction, so use dummy drivers
# to make inspection safe in headless terminal environments.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from pettingzoo.sisl import pursuit_v4


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts"
OUT_PATH = OUT_DIR / "pursuit_v4_inspection.json"


def to_serializable(value: Any) -> Any:
    """Convert numpy-heavy environment outputs into JSON-safe structures."""
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": float(np.min(value)) if value.size else None,
            "max": float(np.max(value)) if value.size else None,
            "sample": value.flatten()[:12].tolist(),
        }
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    return value


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env = pursuit_v4.parallel_env(
        n_pursuers=3,
        n_evaders=1,
        obs_range=7,
        render_mode=None,
    )

    observations, infos = env.reset(seed=0)
    possible_agents = list(env.possible_agents)
    action_spaces = {
        agent: str(env.action_space(agent)) for agent in possible_agents
    }
    observation_spaces = {
        agent: str(env.observation_space(agent)) for agent in possible_agents
    }

    first_agent = possible_agents[0]
    first_obs = observations[first_agent]

    random_actions = {
        agent: env.action_space(agent).sample()
        for agent in env.agents
    }
    next_observations, rewards, terminations, truncations, next_infos = env.step(
        random_actions
    )

    summary = {
        "env": "pettingzoo.sisl.pursuit_v4.parallel_env",
        "config": {
            "n_pursuers": 3,
            "n_evaders": 1,
            "obs_range": 7,
            "render_mode": None,
            "seed": 0,
        },
        "possible_agents": possible_agents,
        "active_agents_after_reset": list(env.agents),
        "action_spaces": action_spaces,
        "observation_spaces": observation_spaces,
        "reset_info_keys": {agent: sorted(info.keys()) for agent, info in infos.items()},
        "first_observation_summary": {
            "agent": first_agent,
            "python_type": type(first_obs).__name__,
            "value": to_serializable(first_obs),
        },
        "sampled_actions": to_serializable(random_actions),
        "step_outputs": {
            "next_observation_keys": list(next_observations.keys()),
            "rewards": to_serializable(rewards),
            "terminations": to_serializable(terminations),
            "truncations": to_serializable(truncations),
            "info_keys": {
                agent: sorted(info.keys()) for agent, info in next_infos.items()
            },
        },
    }

    OUT_PATH.write_text(json.dumps(summary, indent=2))

    print(f"Wrote inspection artifact to {OUT_PATH}")
    print("Possible agents:", possible_agents)
    print("Active agents after reset:", env.agents)
    print("Action space for first agent:", action_spaces[first_agent])
    print("Observation space for first agent:", observation_spaces[first_agent])
    print("First observation summary:")
    print(json.dumps(summary["first_observation_summary"], indent=2))

    env.close()


if __name__ == "__main__":
    main()
