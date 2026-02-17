"""
Configuration for Tabular Q-Learning on MiniGrid Environment
=============================================================
"""

# Environment Configuration
ENV_CONFIG = {
    "size": 12,                    # 10x10 playable area (12x12 with border walls)
    "wall_length": 4,              # Length of wall obstacle
    "wall_position_range": (0.3, 0.7),  # Wall placement between 30%-70% of path
    "max_steps": 120,              # Maximum steps per episode (reduced from 200)
}

# Q-Learning Hyperparameters
AGENT_CONFIG = {
    "n_actions": 3,                # Actions: 0=turn_left, 1=turn_right, 2=move_forward
    "learning_rate": 0.1,          # Alpha: learning rate
    "discount_factor": 0.99,       # Gamma: discount factor
    "epsilon": 1.0,                # Initial exploration rate
    "epsilon_delay": 1350,
    "epsilon_decay": 0.9995,        # Epsilon decay per episode
    "epsilon_min": 0.05,           # Minimum epsilon
    "distance_reward_scale": 0.05,
    "coord_clamp": 4,
    "wall_clamp": 1  # Coarse wall vector: -1, 0, +1 (for CustomAgent11)
}

# Training Configuration
TRAIN_CONFIG = {
    "n_episodes": 5000,            # Total training episodes
    "eval_interval": 500,          # Evaluate every N episodes
    "save_interval": 1000,         # Save checkpoint every N episodes
    "render_training": False,      # Render during training (slow)
    "log_interval": 100,           # Print stats every N episodes
}

# Evaluation Configuration
EVAL_CONFIG = {
    "n_episodes": 100,             # Number of episodes for evaluation
    "render": True,                # Render evaluation episodes
    "delay": 0.1,                  # Delay between steps when rendering
}

# Paths
PATHS = {
    "save_dir": "./checkpoints",
    "logs_dir": "./logs",
    "q_table_file": "./checkpoints/q_table.pkl",
    # Versioned checkpoints for easy evaluation:
    # python main.py eval --checkpoint checkpoints/v3.pkl --no-render
    "v2_failed": "./checkpoints/q_table_v2_FAILED_wandering_agent.pkl",
    "v3_file": "./checkpoints/q_table_ep5000.pkl",  # Current best
}
