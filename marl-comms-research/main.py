"""Main entry point for the MARL communication research workspace."""

from __future__ import annotations

import argparse
import os
import time

from envs import MAP_SPECS, make_fixed_pursuit_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research entry point for the MARL communication project."
    )
    parser.add_argument(
        "--map",
        default="center_block",
        choices=sorted(MAP_SPECS),
        help="Fixed map preset to run.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Maximum number of steps to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used at reset.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=6.0,
        help="Playback speed when rendering in a window.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening a window.",
    )
    parser.add_argument(
        "--freeze-prey",
        action="store_true",
        help="Freeze the prey in place for simple debugging runs.",
    )
    return parser.parse_args()


def configure_runtime(headless: bool) -> None:
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def main() -> None:
    args = parse_args()
    configure_runtime(headless=args.headless)

    render_mode = None if args.headless else "human"
    env = make_fixed_pursuit_env(
        map_name=args.map,
        render_mode=render_mode,
        freeze_evaders=args.freeze_prey,
    )

    observations, infos = env.reset(seed=args.seed)
    first_agent = env.possible_agents[0]

    print("Starting MARL research environment")
    print(f"Map: {args.map}")
    print(f"Agents: {env.agents}")
    print(f"Action space: {env.action_space(first_agent)}")
    print(f"Observation shape: {observations[first_agent].shape}")
    print(f"Initial info for {first_agent}: {infos[first_agent]}")

    total_rewards = {agent: 0.0 for agent in env.possible_agents}

    try:
        for step in range(args.steps):
            if not env.agents:
                print(f"Episode ended early at step {step}.")
                break

            actions = {
                agent: env.action_space(agent).sample()
                for agent in env.agents
            }
            observations, rewards, terminations, truncations, infos = env.step(actions)
            del observations

            for agent, reward in rewards.items():
                total_rewards[agent] += reward

            if render_mode == "human":
                env.render()
                time.sleep(max(0.0, 1.0 / args.fps))

            if any(terminations.values()) or any(truncations.values()):
                print(f"Episode finished at step {step + 1}.")
                print(f"Sample terminal info: {infos[first_agent]}")
                break
        else:
            print(f"Reached step limit of {args.steps}.")
    finally:
        env.close()

    print("Total rewards:")
    for agent, reward in total_rewards.items():
        print(f"  {agent}: {reward:.3f}")


if __name__ == "__main__":
    main()
