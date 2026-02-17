"""
Test Script for GridWorld Environment
======================================
Simple script to verify the environment works correctly.

Usage:
    python test_env.py
    python test_env.py --episodes 10 --render

Author: Chenna (CS Senior, Bethune-Cookman University)
"""

from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, str(__file__).rsplit("/", 1)[0] if "/" in __file__ else ".")

from envs.grid_world_env import GridWorldEnv


def test_environment(episodes: int = 5, render: bool = False, delay: float = 0.2):
    """
    Run a simple test on the GridWorld environment.

    Args:
        episodes: Number of test episodes to run.
        render: Whether to render the environment.
        delay: Delay between steps when rendering.
    """
    print("=" * 60)
    print("GRIDWORLD ENVIRONMENT TEST")
    print("=" * 60)

    # Create environment
    env = GridWorldEnv(
        size=12,
        max_steps=100,
        wall_length=4,
        render_mode="human" if render else None,
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
            # Print ASCII representation
            print("\n" + env._render_text())

        # Random action loop
        while not done:
            # Random action
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

    # Summary
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

    # Test reachability check
    print("\n" + "=" * 60)
    print("REACHABILITY TEST")
    print("=" * 60)
    test_reachability(env)


def test_reachability(env: GridWorldEnv, n_tests: int = 10):
    """Test that the reachability check works correctly."""
    reachable_count = 0

    for i in range(n_tests):
        env.reset()
        is_reachable = env._is_reachable(env.agent_pos, env.goal_pos)
        if is_reachable:
            reachable_count += 1
        print(f"  Test {i+1}: Goal at {env.goal_pos} is {'reachable' if is_reachable else 'NOT reachable'} from {env.agent_pos}")

    print(f"\n  All {n_tests} generated grids have reachable goals: {'PASS' if reachable_count == n_tests else 'FAIL'}")


def test_action_deltas():
    """Test that action deltas are correct."""
    print("\n" + "=" * 60)
    print("ACTION DELTA TEST")
    print("=" * 60)

    # Create a clean grid for testing (no walls except border)
    env = GridWorldEnv(size=7)  # 5x5 playable area
    env.reset(seed=42)
    
    # Clear all obstacles and internal walls for this test
    env.grid.fill(env.TILE_EMPTY)
    env.grid[0, :] = env.TILE_WALL  # Top border
    env.grid[-1, :] = env.TILE_WALL  # Bottom border
    env.grid[:, 0] = env.TILE_WALL  # Left border
    env.grid[:, -1] = env.TILE_WALL  # Right border
    env.obstacle_positions = []

    # Test from center position (3, 3) in a 7x7 grid
    test_cases = [
        (GridWorldEnv.UP, (3, 2), "UP should decrease y"),
        (GridWorldEnv.DOWN, (3, 4), "DOWN should increase y"),
        (GridWorldEnv.LEFT, (2, 3), "LEFT should decrease x"),
        (GridWorldEnv.RIGHT, (4, 3), "RIGHT should increase x"),
    ]

    all_passed = True
    for action, expected_pos, description in test_cases:
        env.agent_pos = (3, 3)  # Reset position to center
        obs, _, _, _, _ = env.step(action)
        actual_pos = (int(obs['agent_pos'][0]), int(obs['agent_pos'][1]))
        passed = actual_pos == expected_pos
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {description}")
        print(f"       Expected: {expected_pos}, Got: {actual_pos}")

    print(f"\n  All action tests: {'PASS' if all_passed else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser(
        description="Test the GridWorld environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_env.py              # Run tests without rendering
  python test_env.py --render     # Run tests with visual rendering
  python test_env.py --action-test  # Test action deltas only
        """,
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of test episodes (default: 5)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment during tests",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between steps when rendering (default: 0.1)",
    )
    parser.add_argument(
        "--action-test",
        action="store_true",
        help="Only run action delta tests",
    )

    args = parser.parse_args()

    if args.action_test:
        test_action_deltas()
    else:
        test_environment(args.episodes, args.render, args.delay)
        test_action_deltas()


if __name__ == "__main__":
    main()
