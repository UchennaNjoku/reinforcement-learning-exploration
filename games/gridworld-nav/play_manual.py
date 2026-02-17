"""
Manual Control for GridWorld Navigation
========================================
Play the GridWorld environment manually using keyboard controls.

Controls:
    W / UP arrow    = Move UP
    S / DOWN arrow  = Move DOWN
    A / LEFT arrow  = Move LEFT
    D / RIGHT arrow = Move RIGHT
    R               = Reset environment
    ESC / Q         = Quit

Usage:
    python play_manual.py
    python play_manual.py --size 12
    python play_manual.py --help

Author: Chenna (CS Senior, Bethune-Cookman University)
"""

from __future__ import annotations

import argparse
import sys

import gymnasium as gym
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(__file__).rsplit("/", 1)[0] if "/" in __file__ else ".")

from envs.grid_world_env import GridWorldEnv


def play_manual(size: int = 12, max_steps: int = 120, wall_length: int = 4):
    """
    Run the GridWorld environment with manual keyboard control.

    Args:
        size: Grid size (including border walls).
        max_steps: Maximum steps per episode.
        wall_length: Length of the blocking wall obstacle.
    """
    try:
        import pygame
    except ImportError:
        print("Error: pygame is required for manual control.")
        print("Install with: pip install pygame")
        sys.exit(1)

    # Create environment
    env = GridWorldEnv(
        size=size,
        max_steps=max_steps,
        wall_length=wall_length,
        render_mode="human",
    )

    # Action mapping from keys
    action_map = {
        pygame.K_UP: GridWorldEnv.UP,
        pygame.K_w: GridWorldEnv.UP,
        pygame.K_DOWN: GridWorldEnv.DOWN,
        pygame.K_s: GridWorldEnv.DOWN,
        pygame.K_LEFT: GridWorldEnv.LEFT,
        pygame.K_a: GridWorldEnv.LEFT,
        pygame.K_RIGHT: GridWorldEnv.RIGHT,
        pygame.K_d: GridWorldEnv.RIGHT,
    }

    print("\n" + "=" * 60)
    print("GRIDWORLD NAVIGATION - MANUAL CONTROL")
    print("=" * 60)
    print("Controls:")
    print("  W / UP    = Move UP")
    print("  S / DOWN  = Move DOWN")
    print("  A / LEFT  = Move LEFT")
    print("  D / RIGHT = Move RIGHT")
    print("  R         = Reset")
    print("  ESC / Q   = Quit")
    print("=" * 60 + "\n")

    # Reset environment
    obs, info = env.reset()
    
    print(f"Agent starts at: {obs['agent_pos']}")
    print(f"Goal is at:      {obs['goal_pos']}")
    print(f"Wall tiles:      {len(obs['obstacle_positions'])}")
    print(f"Distance to goal: {info['dist_to_goal']:.2f}")
    print()

    running = True
    episode_reward = 0.0
    step_count = 0

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            if event.type == pygame.KEYDOWN:
                # Quit
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                    break

                # Reset
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    episode_reward = 0.0
                    step_count = 0
                    print("\n--- Environment Reset ---")
                    print(f"Agent starts at: {obs['agent_pos']}")
                    print(f"Goal is at:      {obs['goal_pos']}")
                    print(f"Wall tiles:      {len(obs['obstacle_positions'])}")
                    print(f"Distance to goal: {info['dist_to_goal']:.2f}")
                    print()
                    continue

                # Movement
                if event.key in action_map:
                    action = action_map[event.key]
                    action_name = ["UP", "DOWN", "LEFT", "RIGHT"][action]

                    # Take step
                    obs, reward, terminated, truncated, info = env.step(action)
                    step_count += 1
                    episode_reward += reward

                    # Print status
                    status = f"Step {step_count}: {action_name} | "
                    status += f"Agent at {tuple(obs['agent_pos'])} | "
                    status += f"Reward: {reward:.2f}"
                    
                    if terminated:
                        status += " | GOAL REACHED!"
                    elif truncated:
                        status += " | MAX STEPS REACHED"
                    
                    print(status)

                    # Check episode end
                    if terminated or truncated:
                        print(f"\nEpisode finished!")
                        print(f"Total reward: {episode_reward:.2f}")
                        print(f"Total steps: {step_count}")
                        print(f"Success: {'YES' if terminated else 'NO'}")
                        print("\nPress R to reset, ESC to quit\n")

    env.close()
    print("\nGoodbye!")


def play_text_mode(size: int = 12, max_steps: int = 120, wall_length: int = 4):
    """
    Play in text mode (no pygame required).
    Uses input() for commands.
    """
    env = GridWorldEnv(
        size=size,
        max_steps=max_steps,
        wall_length=wall_length,
        render_mode="ansi",
    )

    action_map = {
        'w': GridWorldEnv.UP,
        'up': GridWorldEnv.UP,
        's': GridWorldEnv.DOWN,
        'down': GridWorldEnv.DOWN,
        'a': GridWorldEnv.LEFT,
        'left': GridWorldEnv.LEFT,
        'd': GridWorldEnv.RIGHT,
        'right': GridWorldEnv.RIGHT,
    }

    print("\n" + "=" * 60)
    print("GRIDWORLD NAVIGATION - TEXT MODE")
    print("=" * 60)
    print("Commands: w/up, s/down, a/left, d/right, r/reset, q/quit")
    print("=" * 60 + "\n")

    obs, info = env.reset()
    episode_reward = 0.0
    step_count = 0

    print(env.render())
    print(f"\nAgent: {obs['agent_pos']}, Goal: {obs['goal_pos']}")
    print(f"Distance: {info['dist_to_goal']:.2f}")

    while True:
        cmd = input("\nAction: ").strip().lower()

        if cmd in ('q', 'quit', 'exit'):
            break

        if cmd in ('r', 'reset'):
            obs, info = env.reset()
            episode_reward = 0.0
            step_count = 0
            print("\n--- Reset ---")
            print(env.render())
            continue

        if cmd not in action_map:
            print("Invalid command. Use: w/up, s/down, a/left, d/right, r/reset, q/quit")
            continue

        action = action_map[cmd]
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        episode_reward += reward

        print(env.render())
        print(f"\nAgent: {obs['agent_pos']}, Goal: {obs['goal_pos']}")
        print(f"Step {step_count}, Reward: {reward:.2f}, Total: {episode_reward:.2f}")

        if terminated:
            print("\n*** GOAL REACHED! ***")
            print(f"Success in {step_count} steps!")
        elif truncated:
            print("\n*** MAX STEPS REACHED ***")
            print("Episode truncated.")

    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(
        description="Manual control for GridWorld Navigation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python play_manual.py              # Play with default settings (12x12 grid)
  python play_manual.py --size 16    # Play on a larger 16x16 grid
  python play_manual.py --text       # Play in text mode (no pygame)
        """,
    )
    parser.add_argument(
        "--size",
        type=int,
        default=12,
        help="Grid size including border walls (default: 12, gives 10x10 playable)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=120,
        help="Maximum steps per episode (default: 120)",
    )
    parser.add_argument(
        "--wall-length",
        type=int,
        default=4,
        help="Length of blocking wall obstacle (default: 4)",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Use text mode instead of graphical mode",
    )

    args = parser.parse_args()

    if args.text:
        play_text_mode(args.size, args.max_steps, args.wall_length)
    else:
        play_manual(args.size, args.max_steps, args.wall_length)


if __name__ == "__main__":
    main()
