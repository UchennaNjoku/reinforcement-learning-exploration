"""
Evaluation and visualization script for MiniGrid Wall Navigation.
=================================================================

Evaluate a trained agent or play manually.

Usage:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/q_table_ep5000.pkl --episodes 10
    python evaluate.py --manual          # Play yourself with arrow keys
"""

from __future__ import annotations

import argparse
import time

import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper

from envs.random_goal_obstacle_env import RandomGoalObstacleEnv, SimpleActionWrapper
from agents.custom_agent_13 import CustomAgent13
from config import ENV_CONFIG, PATHS, EVAL_CONFIG


def evaluate_agent(
    checkpoint_path: str | None = None,
    n_episodes: int | None = None,
    render: bool = True,
    delay: float = 0.1,
):
    """
    Evaluate the distance-aware Q-learning agent.

    Args:
        checkpoint_path: Path to saved Q-table.
        n_episodes: Number of evaluation episodes.
        render: Whether to render episodes.
        delay: Delay between steps when rendering.
    """
    checkpoint_path = checkpoint_path or PATHS["q_table_file"]
    n_episodes = n_episodes or EVAL_CONFIG["n_episodes"]
    # Only use config default if render not explicitly set (None means not provided)
    if render is None:
        render = EVAL_CONFIG["render"]

    # Create environment
    env = RandomGoalObstacleEnv(
        size=ENV_CONFIG["size"],
        max_steps=ENV_CONFIG["max_steps"],
        wall_length=ENV_CONFIG["wall_length"],
        wall_position_range=ENV_CONFIG["wall_position_range"],
        render_mode="human" if render else None,
    )
    env = SimpleActionWrapper(env)  # Restrict to 3 navigation actions
    
    if render:
        env = FullyObsWrapper(env)

    # Load agent
    agent = CustomAgent13()
    try:
        agent.load(checkpoint_path)
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Train the agent first with: python train.py")
        return
    
    # FORCE PURE EXPLOITATION - no random actions during evaluation
    agent.epsilon = 0.0
    print(f"  FORCING epsilon = 0.0 (pure greedy policy)")

    print("\n" + "=" * 60)
    print("EVALUATION (Distance-Aware Q-Learning)")
    print("=" * 60)
    print(f"Episodes: {n_episodes}")
    print(f"Epsilon during eval: 0 (greedy policy)")
    print(f"State representation: (dir, dxg, dyg, front_blocked, left_blocked, right_blocked)")
    print(f"  dxg/dyg = goal relative (±4), tri-directional wall sensors")
    print(f"  front_blocked = 1 if tile ahead is wall")
    print(f"Reward: DENSE distance-based shaping + penalties")

    # Evaluation loop
    total_rewards = []
    total_lengths = []
    successes = 0

    for episode in range(1, n_episodes + 1):
        obs, info = env.reset()
        
        # Build state with obstacle positions
        custom_obs = info.get("custom_obs", {})
        agent_pos = tuple(custom_obs.get("agent_pos", env.unwrapped.agent_pos))
        goal_pos = tuple(custom_obs.get("goal_pos", env.unwrapped.goal_pos))
        agent_dir = custom_obs.get("agent_dir", 0)
        dist_to_goal = custom_obs.get("dist_to_goal", 0.0)
        dist_to_obstacle = custom_obs.get("dist_to_obstacle", 0.0)
        obstacle_positions = custom_obs.get("obstacle_positions", [])
        
        state = agent.build_state(agent_pos, goal_pos, agent_dir, dist_to_goal, dist_to_obstacle, obstacle_positions)
        
        episode_reward = 0
        episode_length = 0
        goal_reached = False
        done = False

        print(f"\nEpisode {episode}/{n_episodes}")
        print(f"  Agent: {env.unwrapped.agent_pos}, Goal: {env.unwrapped.goal_pos}")
        print(f"  Wall tiles: {len(env.unwrapped.obstacle_positions)}")
        print(f"  Dist to goal: {dist_to_goal:.2f}, Dist to obstacle: {dist_to_obstacle:.2f}")

        while not done:
            if render:
                env.render()
                time.sleep(delay)

            # Select action (no exploration - greedy only)
            # Extract front_blocked from state for action masking (CustomAgent13)
            front_blocked = state[5] if len(state) >= 6 else 0
            action = agent.get_action(state, training=False, front_blocked=front_blocked)
            
            # Remember distance before action (for dense reward)
            prev_dist_to_goal = dist_to_goal
            prev_agent_pos = agent_pos
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Apply SAME reward shaping as training
            # Potential-based shaping: F = scale * (gamma*Phi(s') - Phi(s)), Phi(s) = -dist
            custom_obs_temp = info.get("custom_obs", {})
            new_dist_to_goal = custom_obs_temp.get("dist_to_goal", dist_to_goal)
            phi_s = -prev_dist_to_goal
            phi_sp = -new_dist_to_goal
            reward += agent.distance_reward_scale * (agent.discount_factor * phi_sp - phi_s)
            
            # 2. Step penalty
            reward -= 0.01
            
            # 3. Tiny penalty for turning (prevents spinning loops)
            if action in (0, 1):  # turn_left or turn_right
                reward -= 0.01
            
            # 4. Wall collision penalty
            if action == 2:  # move_forward
                new_pos = tuple(custom_obs_temp.get("agent_pos", env.unwrapped.agent_pos))
                if new_pos == prev_agent_pos:
                    reward -= 0.05  # Wall bump penalty
            
            # Build next state with obstacle positions
            custom_obs = info.get("custom_obs", {})
            next_agent_pos = tuple(custom_obs.get("agent_pos", env.unwrapped.agent_pos))
            next_goal_pos = tuple(custom_obs.get("goal_pos", env.unwrapped.goal_pos))
            next_agent_dir = custom_obs.get("agent_dir", 0)
            next_dist_to_goal = custom_obs.get("dist_to_goal", 0.0)
            next_dist_to_obstacle = custom_obs.get("dist_to_obstacle", 0.0)
            next_obstacle_positions = custom_obs.get("obstacle_positions", [])
            
            next_state = agent.build_state(
                next_agent_pos, next_goal_pos, next_agent_dir,
                next_dist_to_goal, next_dist_to_obstacle, next_obstacle_positions
            )
            
            done = terminated or truncated
            
            # Track if goal was reached (for success metric)
            if terminated:
                goal_reached = True
            
            # 5. Timeout penalty
            if truncated and not terminated:
                reward -= 1.0
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            agent_pos = next_agent_pos
            dist_to_goal = next_dist_to_goal

        total_rewards.append(episode_reward)
        total_lengths.append(episode_length)
        
        success = goal_reached  # Success = agent reached the goal
        if success:
            successes += 1
        
        print(f"  Result: {'SUCCESS' if success else 'FAILED'} | "
              f"Reward: {episode_reward:.2f} | Steps: {episode_length}")

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Success rate: {successes}/{n_episodes} ({100*successes/n_episodes:.1f}%)")
    print(f"Avg reward: {sum(total_rewards)/len(total_rewards):.2f}")
    print(f"Avg steps: {sum(total_lengths)/len(total_lengths):.1f}")
    print(f"Min steps: {min(total_lengths)}")
    print(f"Max steps: {max(total_lengths)}")

    env.close()


def manual_control():
    """Run the environment with manual keyboard control."""
    from minigrid.manual_control import ManualControl

    print("\n" + "=" * 60)
    print("MANUAL CONTROL MODE")
    print("=" * 60)
    print("Controls:")
    print("  Arrow keys = move/turn")
    print("  R = reset")
    print("  Esc = quit")
    print("=" * 60 + "\n")

    env = RandomGoalObstacleEnv(
        size=ENV_CONFIG["size"],
        wall_length=ENV_CONFIG["wall_length"],
        render_mode="human",
    )
    env = SimpleActionWrapper(env)  # Restrict to 3 navigation actions
    env = FullyObsWrapper(env)

    obs, info = env.reset()
    custom = info.get("custom_obs", {})
    
    print(f"Agent starts at: {custom.get('agent_pos')}")
    print(f"Goal is at:      {custom.get('goal_pos')}")
    print(f"Wall tiles:      {len(custom.get('obstacle_positions', []))}")
    print(f"Distance to goal: {custom.get('dist_to_goal', 0):.2f}")

    manual_control = ManualControl(env, seed=None)
    manual_control.start()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Q-learning agent")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to Q-table checkpoint",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Delay between steps (seconds)",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Run in manual control mode",
    )

    args = parser.parse_args()

    if args.manual:
        manual_control()
    else:
        evaluate_agent(
            checkpoint_path=args.checkpoint,
            n_episodes=args.episodes,
            render=not args.no_render,
            delay=args.delay or EVAL_CONFIG["delay"],
        )


if __name__ == "__main__":
    main()
