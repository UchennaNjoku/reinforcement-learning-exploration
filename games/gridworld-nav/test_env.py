from __future__ import annotations

import argparse
import time
from env import GridWorldEnv

from environment.constants import TILE_EMPTY, TILE_WALL, UP, DOWN, LEFT, RIGHT
from environment.generation import bfs_reachable


def test_environment(episodes: int = 5, render: bool = False, delay: float = 0.2):
    print("=" * 60)
    print("GRIDWORLD ENVIRONMENT TEST")
    print("=" * 60)

    env = GridWorldEnv(
        size=12,
        max_steps=100,
        wall_length=4,
        render_mode="human" if render else "ansi",   # ✅ important
    )

    print(f"\nEnvironment Info:")
    print(f"  Grid size: {env.size}x{env.size} (including border)")
    print(f"  Playable area: {env.playable_size}x{env.playable_size}")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Max steps per episode: {env.max_steps}")
    print(f"  Actions: UP=0, DOWN=1, LEFT=2, RIGHT=3")

    total_rewards = []
    total_steps = []
    successes = 0

    for episode in range(1, episodes + 1):
        obs, info = env.reset()
        episode_reward = 0.0
        step_count = 0
        done = False

        print(f"\n--- Episode {episode}/{episodes} ---")
        print(f"  Agent start: {obs['agent_pos']}")
        print(f"  Goal:        {obs['goal_pos']}")
        print(f"  Wall tiles:  {len(obs['obstacle_positions'])}")
        print(f"  Distance:    {info['dist_to_goal']:.2f}")

        if not render:
            print("\n" + env.render())  # ✅ ansi render string

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            done = terminated or truncated

            if render:
                time.sleep(delay)

        total_rewards.append(episode_reward)
        total_steps.append(step_count)

        if terminated:
            successes += 1
            result = "SUCCESS"
        else:
            result = "FAILED"

        print(f"  Result: {result}")
        print(f"  Steps: {step_count}")
        print(f"  Reward: {episode_reward:.2f}")

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Episodes: {episodes}")
    print(f"Successes: {successes}/{episodes} ({100*successes/episodes:.1f}%)")
    print(f"Avg reward: {sum(total_rewards)/len(total_rewards):.2f}")
    print(f"Avg steps: {sum(total_steps)/len(total_steps):.1f}")
    print(f"Min steps: {min(total_steps)}")
    print(f"Max steps: {max(total_steps)}")

    env.close()

    print("\n" + "=" * 60)
    print("REACHABILITY TEST")
    print("=" * 60)
    test_reachability(env)


def test_reachability(env: GridWorldEnv, n_tests: int = 10):
    reachable_count = 0
    for i in range(n_tests):
        env.reset()
        ok = bfs_reachable(env.grid, env.size, env.agent_pos, env.goal_pos)  # ✅ moved to generation.py
        reachable_count += int(ok)
        print(f"  Test {i+1}: Goal at {env.goal_pos} is {'reachable' if ok else 'NOT reachable'} from {env.agent_pos}")

    print(f"\n  All {n_tests} generated grids have reachable goals: {'PASS' if reachable_count == n_tests else 'FAIL'}")


def test_action_deltas():
    print("\n" + "=" * 60)
    print("ACTION DELTA TEST")
    print("=" * 60)

    env = GridWorldEnv(size=7)  # 5x5 playable area
    env.reset(seed=42)

    env.grid.fill(TILE_EMPTY)
    env.grid[0, :] = TILE_WALL
    env.grid[-1, :] = TILE_WALL
    env.grid[:, 0] = TILE_WALL
    env.grid[:, -1] = TILE_WALL
    env.obstacle_positions = []

    test_cases = [
        (UP, (3, 2), "UP should decrease y"),
        (DOWN, (3, 4), "DOWN should increase y"),
        (LEFT, (2, 3), "LEFT should decrease x"),
        (RIGHT, (4, 3), "RIGHT should increase x"),
    ]

    all_passed = True
    for action, expected_pos, description in test_cases:
        env.agent_pos = (3, 3)
        obs, _, _, _, _ = env.step(action)
        actual_pos = (int(obs["agent_pos"][0]), int(obs["agent_pos"][1]))
        passed = actual_pos == expected_pos
        all_passed = all_passed and passed
        print(f"  {'PASS' if passed else 'FAIL'}: {description}")
        print(f"       Expected: {expected_pos}, Got: {actual_pos}")

    print(f"\n  All action tests: {'PASS' if all_passed else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser(
        description="Test the GridWorld environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--action-test", action="store_true")
    args = parser.parse_args()

    if args.action_test:
        test_action_deltas()
    else:
        test_environment(args.episodes, args.render, args.delay)
        test_action_deltas()


if __name__ == "__main__":
    main()