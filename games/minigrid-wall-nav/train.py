"""
Training script for Tabular Q-Learning on MiniGrid Wall Navigation.
===================================================================

Train a Q-learning agent to navigate around wall obstacles to reach a goal.

Usage:
    python train.py
    python train.py --episodes 10000 --render
    python train.py --agent CustomAgent11 --episodes 15000
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime

import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper

from envs.random_goal_obstacle_env import RandomGoalObstacleEnv, SimpleActionWrapper
from agents.custom_agent import CustomAgent
from agents.custom_agent_2 import CustomAgent2
from agents.custom_agent_3 import CustomAgent3
from agents.custom_agent_4 import CustomAgent4
from agents.custom_agent_5 import CustomAgent5
from agents.custom_agent_6 import CustomAgent6
from agents.custom_agent_7 import CustomAgent7
from agents.custom_agent_8 import CustomAgent8
from agents.custom_agent_9 import CustomAgent9
from agents.custom_agent_10 import CustomAgent10
from agents.custom_agent_11 import CustomAgent11
from agents.custom_agent_12 import CustomAgent12
from agents.custom_agent_13 import CustomAgent13
from config import ENV_CONFIG, AGENT_CONFIG, TRAIN_CONFIG, PATHS
from utils import setup_directories, plot_training_stats, print_episode_info


# Map agent names to classes
AGENT_CLASSES = {
    "CustomAgent": CustomAgent,
    "CustomAgent2": CustomAgent2,
    "CustomAgent3": CustomAgent3,
    "CustomAgent4": CustomAgent4,
    "CustomAgent5": CustomAgent5,
    "CustomAgent6": CustomAgent6,
    "CustomAgent7": CustomAgent7,
    "CustomAgent8": CustomAgent8,
    "CustomAgent9": CustomAgent9,
    "CustomAgent10": CustomAgent10,
    "CustomAgent11": CustomAgent11,
    "CustomAgent12": CustomAgent12,
    "CustomAgent13": CustomAgent13,
}


def train_agent(
    n_episodes: int | None = None,
    render: bool = False,
    save_path: str | None = None,
    agent_class: type | None = None,
    epsilon_delay: int | None = None,
    epsilon_decay: float | None = None,
    curriculum: bool = False,
):
    """
    Train the distance-aware Q-learning agent.

    Args:
        n_episodes: Number of training episodes (default from config).
        render: Whether to render training episodes.
        save_path: Path to save the final Q-table.
    """
    # Setup
    setup_directories(PATHS)
    
    n_episodes = n_episodes or TRAIN_CONFIG["n_episodes"]
    save_path = save_path or PATHS["q_table_file"]

    # Curriculum learning setup
    curriculum_phase = "N/A"
    if curriculum:
        print("\n[CURRICULUM LEARNING ENABLED]")
        print("Phase 1 (0-25%): No wall (wall_length=0)")
        print("Phase 2 (25-50%): Small wall (wall_length=2)")
        print("Phase 3 (50%+): Full wall (wall_length=4)")

    # Create environment
    print("Creating environment...")
    env = RandomGoalObstacleEnv(
        size=ENV_CONFIG["size"],
        max_steps=ENV_CONFIG["max_steps"],
        wall_length=ENV_CONFIG["wall_length"],
        wall_position_range=ENV_CONFIG["wall_position_range"],
    )
    env = SimpleActionWrapper(env)  # Restrict to 3 navigation actions
    
    if render:
        env = FullyObsWrapper(gym.wrappers.HumanRendering(env))
    
    print(f"Environment: {ENV_CONFIG['size']}x{ENV_CONFIG['size']} grid")
    print(f"Playable area: {ENV_CONFIG['size']-2}x{ENV_CONFIG['size']-2}")

    # Create agent
    agent_name = agent_class.__name__ if agent_class else "CustomAgent11"
    print(f"\nInitializing {agent_name}...")
    
    # Build agent config with optional overrides
    agent_config = AGENT_CONFIG.copy()
    if epsilon_delay is not None:
        agent_config["epsilon_delay"] = epsilon_delay
    if epsilon_decay is not None:
        agent_config["epsilon_decay"] = epsilon_decay
    
    agent = agent_class(**agent_config) if agent_class else CustomAgent11(**agent_config)
    print(f"Learning rate: {AGENT_CONFIG['learning_rate']}")
    print(f"Discount factor: {AGENT_CONFIG['discount_factor']}")
    print(f"Initial epsilon: {AGENT_CONFIG['epsilon']}")
    print(f"Epsilon delay: {agent.epsilon_delay_episodes} episodes")
    print(f"Epsilon decay: {agent.epsilon_decay}")
    print(f"State features: (dir, dxg, dyg, dxw, dyw, front_blocked, left_blocked, right_blocked)")
    print(f"  dxg/dyg = goal relative (±4, precise)")
    print(f"  dxw/dyw = wall relative (±1, coarse: -1,0,+1)")
    print(f"  front/left/right_blocked = wall sensors")
    print(f"  State space: 4 × 9 × 9 × 3 × 3 × 2 × 2 × 2 = ~23k states")
    print(f"Reward: dense distance-based + -0.01/step + -0.05/bump + -1.0/timeout")
    print(f"Learning: rate=0.5 (fast)")
    if curriculum:
        print(f"Curriculum: Progressive difficulty with smooth epsilon decay")
        print(f"  Phase 1 (0-25%): eps=1.0 → 0.15, decay=0.995 (learn pure goal-chasing)")
        print(f"  Phase 2 (25-50%): eps continues → 0.10, decay=0.999 (learn detour)")
        print(f"  Phase 3 (50%+): eps continues → 0.05, decay=0.9995 (fine-tune)")
    else:
        print(f"Exploration: epsilon=1.0 for first {agent.epsilon_delay_episodes} eps, then decay={agent.epsilon_decay:.4f}, min={agent.epsilon_min}")

    # Training loop
    print(f"\nStarting training for {n_episodes} episodes...")
    print("=" * 80)
    
    start_time = time.time()
    recent_rewards = []

    for episode in range(1, n_episodes + 1):
        # Curriculum: Adjust wall difficulty based on episode progress
        if curriculum:
            progress = episode / n_episodes
            if progress < 0.25:
                env.unwrapped.wall_length = 0
                curriculum_phase = "Phase 1 (no wall)"
            elif progress < 0.50:
                env.unwrapped.wall_length = 2
                curriculum_phase = "Phase 2 (small wall)"
            else:
                env.unwrapped.wall_length = ENV_CONFIG["wall_length"]  # Full difficulty
                curriculum_phase = "Phase 3 (full wall)"
        
        obs, info = env.reset()
        
        # Extract distance information from environment info
        custom_obs = info.get("custom_obs", {})
        agent_pos = tuple(custom_obs.get("agent_pos", env.unwrapped.agent_pos))
        goal_pos = tuple(custom_obs.get("goal_pos", env.unwrapped.goal_pos))
        agent_dir = custom_obs.get("agent_dir", 0)
        dist_to_goal = custom_obs.get("dist_to_goal", 0.0)
        dist_to_obstacle = custom_obs.get("dist_to_obstacle", 0.0)
        
        # Build state with obstacle positions (NEW)
        obstacle_positions = custom_obs.get("obstacle_positions", [])
        state = agent.build_state(agent_pos, goal_pos, agent_dir, dist_to_goal, dist_to_obstacle, obstacle_positions)
        
        episode_reward = 0
        episode_length = 0
        episode_q_change = 0
        done = False

        while not done:
            if render:
                env.render()
                time.sleep(0.05)

            # Select and perform action
            # Extract front_blocked from state for action masking (CustomAgent13)
            front_blocked = state[5] if len(state) >= 6 else 0
            action = agent.get_action(state, training=True, front_blocked=front_blocked)
            
            # Remember position and distance before action
            prev_agent_pos = agent_pos
            prev_dist_to_goal = dist_to_goal  # For dense reward
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Reward shaping v7: DENSE reward based on progress toward goal
            # Actions: 0=turn_left, 1=turn_right, 2=move_forward
            
            # Get new distance after action
            custom_obs_temp = info.get("custom_obs", {})
            new_dist_to_goal = custom_obs_temp.get("dist_to_goal", dist_to_goal)
            
            # REWARD SHAPING PROVED INEFFECTIVE
            
            # # DENSE REWARD: Reward progress toward goal
            # # If we got closer: positive reward
            # # If we got farther: negative reward
            # distance_improvement = prev_dist_to_goal - new_dist_to_goal
            # reward += distance_improvement * 0.1  # Scale factor
            
            # potential-based shaping: F = scale * (gamma*Phi(s') - Phi(s)), Phi(s) = -dist
            phi_s = -prev_dist_to_goal
            phi_sp = -new_dist_to_goal
            reward += agent.distance_reward_scale * (agent.discount_factor * phi_sp - phi_s)

            # Small step penalty
            reward -= 0.01
            
            # Tiny penalty for turning (prevents spinning loops)
            # Actions: 0=turn_left, 1=turn_right, 2=move_forward
            if action in (0, 1):  # turn_left or turn_right
                reward -= 0.01
            
            # Wall collision detection
            if action == 2:  # move_forward
                new_pos = tuple(custom_obs_temp.get("agent_pos", env.unwrapped.agent_pos))
                if new_pos == prev_agent_pos:
                    reward -= 0.05  # Wall bump penalty
            
            # Build next state with updated distance information
            custom_obs = info.get("custom_obs", {})
            next_agent_pos = tuple(custom_obs.get("agent_pos", env.unwrapped.agent_pos))
            next_goal_pos = tuple(custom_obs.get("goal_pos", env.unwrapped.goal_pos))
            next_agent_dir = custom_obs.get("agent_dir", 0)
            next_dist_to_goal = custom_obs.get("dist_to_goal", 0.0)
            next_dist_to_obstacle = custom_obs.get("dist_to_obstacle", 0.0)
            
            # Build next state with obstacle positions (NEW)
            next_obstacle_positions = custom_obs.get("obstacle_positions", [])
            next_state = agent.build_state(
                next_agent_pos, next_goal_pos, 
                next_agent_dir, next_dist_to_goal, next_dist_to_obstacle,
                next_obstacle_positions
            )
            
            done = terminated or truncated
            

            # Timeout penalty: hitting max_steps is a failure
            if truncated and not terminated:
                reward -= 1.0  # Strong penalty for timeout
            
            # Visited-state penalty (for CustomAgent13 and similar)
            # Penalizes revisiting the same position within an episode
            if hasattr(agent, 'check_visited_penalty'):
                reward += agent.check_visited_penalty(next_agent_pos)
            
            # Update Q-table
            q_change = agent.update(state, action, reward, next_state, terminated)
            episode_q_change += q_change
            
            episode_reward += reward
            episode_length += 1


            # FIX: update trackers used by dense reward + collision checks
            agent_pos = next_agent_pos
            goal_pos = next_goal_pos
            agent_dir = next_agent_dir
            dist_to_goal = next_dist_to_goal
            dist_to_obstacle = next_dist_to_obstacle
            state = next_state

        # Decay exploration
        if curriculum:
            # Continuous epsilon decay with phase-dependent minimums
            # No sudden jumps - smooth decay throughout training
            progress = episode / n_episodes
            if progress < 0.25:
                # Phase 1: High exploration, decay slowly to 0.15
                agent.epsilon = max(0.15, 1.0 * (0.995 ** episode))
            elif progress < 0.50:
                # Phase 2: Medium exploration, continue from current epsilon
                agent.epsilon = max(0.10, agent.epsilon * 0.999)
            else:
                # Phase 3: Low exploration, fine-tuning
                agent.epsilon = max(0.05, agent.epsilon * 0.9995)
        else:
            agent.decay_epsilon()
        
        # Reset visited positions at episode end (for CustomAgent13 and similar)
        if hasattr(agent, 'reset_visited'):
            agent.reset_visited()
        
        # Record stats
        agent.episode_rewards.append(episode_reward)
        agent.episode_lengths.append(episode_length)
        recent_rewards.append(episode_reward)

        # Logging
        if episode % TRAIN_CONFIG["log_interval"] == 0:
            q_states = agent.get_q_table_size() // AGENT_CONFIG["n_actions"]
            curriculum_info = f" | {curriculum_phase}" if curriculum else ""
            print_episode_info(
                episode,
                episode_reward,
                episode_length,
                agent.epsilon,
                q_states,
                recent_rewards,
            )
            if curriculum:
                print(f"  -> {curriculum_phase}")

        # Save checkpoint
        if episode % TRAIN_CONFIG["save_interval"] == 0:
            checkpoint_path = f"{PATHS['save_dir']}/q_table_ep{episode}.pkl"
            agent.save(checkpoint_path)

    # Training complete
    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"\nTraining completed in {elapsed:.1f} seconds")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Total states visited: {agent.get_state_count()}")
    
    # Save final model
    agent.save(save_path)
    
    # Also save as versioned checkpoint for easy evaluation
    versioned_path = f"{PATHS['save_dir']}/{agent_name.lower().replace('customagent', 'v')}.pkl"
    agent.save(versioned_path)
    agent.save(versioned_path)
    print(f"Also saved as: {versioned_path}")
    
    # Plot training stats
    plot_path = f"{PATHS['logs_dir']}/training_stats.png"
    plot_training_stats(
        agent.episode_rewards,
        agent.episode_lengths,
        save_path=plot_path,
    )

    env.close()
    return agent


def main():
    parser = argparse.ArgumentParser(description="Train Q-learning agent on MiniGrid")
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render training episodes (slow)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save Q-table",
    )
    parser.add_argument(
        "--epsilon-delay",
        type=int,
        default=None,
        dest="epsilon_delay",
        help="Number of episodes to delay epsilon decay (default: 1350)",
    )
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=None,
        dest="epsilon_decay",
        help="Epsilon decay rate per episode (default: 0.9995)",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable curriculum learning (progressive wall difficulty)",
    )

    args = parser.parse_args()
    
    train_agent(
        n_episodes=args.episodes,
        render=args.render,
        save_path=args.save_path,
        epsilon_delay=args.epsilon_delay,
        epsilon_decay=args.epsilon_decay,
        curriculum=args.curriculum,
    )


if __name__ == "__main__":
    main()
