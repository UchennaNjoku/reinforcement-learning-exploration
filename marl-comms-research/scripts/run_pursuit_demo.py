"""Run a simple pursuit_v4 demo so the environment can be inspected visually."""

from __future__ import annotations

import argparse
import os
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a basic PettingZoo pursuit_v4 demo."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Maximum number of environment steps to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the environment reset.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=6.0,
        help="Sleep-based playback rate for human rendering.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening a window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    from pettingzoo.sisl import pursuit_v4

    render_mode = None if args.headless else "human"
    env = pursuit_v4.parallel_env(
        n_pursuers=3,
        n_evaders=1,
        obs_range=7,
        render_mode=render_mode,
    )

    observations, infos = env.reset(seed=args.seed)
    del observations, infos

    print("Starting pursuit_v4 demo")
    print(f"Agents: {env.agents}")
    print(f"Render mode: {render_mode}")
    print(f"Action space: {env.action_space(env.agents[0])}")
    print(f"Observation space: {env.observation_space(env.agents[0])}")

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
            del observations, infos

            for agent, reward in rewards.items():
                total_rewards[agent] += float(reward)

            if render_mode == "human":
                env.render()
                time.sleep(max(0.0, 1.0 / args.fps))

            if any(terminations.values()) or any(truncations.values()):
                print(f"Episode finished at step {step + 1}.")
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
