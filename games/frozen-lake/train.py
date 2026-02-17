"""
Frozen Lake Q-Learning Training Script
======================================
A clean implementation of tabular Q-learning for the FrozenLake-v1 environment.

This was the first environment explored in this reinforcement learning journey.
Simple, discrete state space with stochastic (slippery=False) dynamics.

Usage:
    python train.py                    # Train and visualize
    python train.py --episodes 5000    # Train with custom episodes
    python train.py --no-visualize     # Train without final visualization
"""

import argparse
import numpy as np
import gymnasium as gym


def greedy_action_with_ties(Q, state):
    """
    Select action with highest Q-value, breaking ties randomly.
    This helps exploration when multiple actions look equally good.
    """
    row = Q[state]
    max_q = np.max(row)
    best_actions = np.flatnonzero(row == max_q)
    return int(np.random.choice(best_actions))


def train(
    episodes: int = 10000,
    max_steps: int = 100,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 1.0,
    epsilon_min: float = 0.1,
    epsilon_decay: float = 0.995,
    map_name: str = "4x4",
    slippery: bool = False,
    visualize: bool = True,
):
    """
    Train a Q-learning agent on Frozen Lake.
    
    Args:
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Initial exploration rate
        epsilon_min: Minimum exploration rate
        epsilon_decay: Epsilon decay per episode
        map_name: Map size ("4x4" or "8x8")
        slippery: Whether ice is slippery (stochastic transitions)
        visualize: Whether to show final visualization
    """
    # Create environment
    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=slippery, render_mode=None)
    
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    
    # Initialize Q-table
    q_table = np.zeros((num_states, num_actions))
    
    rewards_per_episode = []
    
    print(f"Training on FrozenLake-{map_name} (slippery={slippery})")
    print(f"States: {num_states}, Actions: {num_actions}")
    print(f"Episodes: {episodes}, Learning rate: {alpha}, Gamma: {gamma}")
    print("-" * 60)
    
    for episode in range(episodes):
        state, info = env.reset()
        total_reward = 0
        done = False
        
        for step in range(max_steps):
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = greedy_action_with_ties(q_table, state)  # Exploit
            
            # Take action
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Q-learning update
            q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action]
            )
            
            state = new_state
            total_reward += reward
            
            if done:
                break
        
        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_per_episode.append(total_reward)
        
        # Progress report every 1000 episodes
        if (episode + 1) % 1000 == 0:
            recent = np.mean(rewards_per_episode[-1000:])
            print(f"Episode {episode + 1:5d} | Epsilon: {epsilon:.3f} | Recent Avg Reward: {recent:.3f}")
    
    env.close()
    
    # Print final results
    print("-" * 60)
    print("Training Complete!")
    print("\nResults by 1000-episode chunks:")
    for i in range(0, episodes, 1000):
        chunk = rewards_per_episode[i:i+1000]
        print(f"  Episodes {i:5d}-{i+1000:5d}: avg reward = {np.mean(chunk):.3f}")
    
    print(f"\nOverall success rate: {np.mean(rewards_per_episode)*100:.1f}%")
    
    # Save Q-table
    np.save("checkpoints/q_table.npy", q_table)
    print("\nQ-table saved to checkpoints/q_table.npy")
    
    # Visualization
    if visualize:
        print("\nRunning visualization with learned policy...")
        visualize_policy(q_table, map_name, slippery)
    
    return q_table, rewards_per_episode


def visualize_policy(q_table, map_name="4x4", slippery=False):
    """
    Visualize the trained policy.
    """
    env_visual = gym.make(
        "FrozenLake-v1", 
        map_name=map_name, 
        is_slippery=slippery, 
        render_mode="human"
    )
    state, _ = env_visual.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < 100:
        action = np.argmax(q_table[state, :])  # Pure exploitation
        state, reward, terminated, truncated, _ = env_visual.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
    
    if total_reward == 1:
        print(f"✓ Agent reached the goal in {steps} steps!")
    else:
        print(f"✗ Agent failed after {steps} steps")
    
    env_visual.close()


def evaluate(q_table_path: str, map_name: str = "4x4", slippery: bool = False, episodes: int = 100):
    """
    Evaluate a saved Q-table.
    """
    q_table = np.load(q_table_path)
    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=slippery, render_mode=None)
    
    successes = 0
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            action = np.argmax(q_table[state, :])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if reward == 1:
                successes += 1
    
    env.close()
    print(f"Evaluation: {successes}/{episodes} successes ({100*successes/episodes:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Train Q-learning on Frozen Lake")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of training episodes")
    parser.add_argument("--map", type=str, default="4x4", choices=["4x4", "8x8"], help="Map size")
    parser.add_argument("--slippery", action="store_true", help="Enable slippery ice")
    parser.add_argument("--no-visualize", action="store_true", help="Skip visualization")
    parser.add_argument("--evaluate", type=str, default=None, help="Evaluate saved Q-table")
    
    args = parser.parse_args()
    
    if args.evaluate:
        evaluate(args.evaluate, args.map, args.slippery)
    else:
        train(
            episodes=args.episodes,
            map_name=args.map,
            slippery=args.slippery,
            visualize=not args.no_visualize,
        )


if __name__ == "__main__":
    main()
