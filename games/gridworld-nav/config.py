"""
Configuration for GridWorld Navigation
======================================
"""

# Environment Configuration
ENV_CONFIG = {
    "size": 12,                    # 10x10 playable area (12x12 with border walls)
    "wall_length": 4,              # Length of wall obstacle
    "wall_position_range": (0.3, 0.7),  # Wall placement between 30%-70% of path
    "max_steps": 120,              # Maximum steps per episode
}

# Q-Learning Hyperparameters (for future use)
AGENT_CONFIG = {
    "n_actions": 4,                # Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
    "learning_rate": 0.1,          # Alpha: learning rate
    "discount_factor": 0.99,       # Gamma: discount factor
    "epsilon": 1.0,                # Initial exploration rate
    "epsilon_decay": 0.995,        # Epsilon decay per episode
    "epsilon_min": 0.01,           # Minimum epsilon
}

# Training Configuration (for future use)
TRAIN_CONFIG = {
    "n_episodes": 5000,            # Total training episodes
    "eval_interval": 500,          # Evaluate every N episodes
    "save_interval": 1000,         # Save checkpoint every N episodes
    "render_training": False,      # Render during training (slow)
    "log_interval": 100,           # Print stats every N episodes
}

# Paths (for future use)
PATHS = {
    "save_dir": "./checkpoints",
    "logs_dir": "./logs",
    "q_table_file": "./checkpoints/q_table.pkl",
}
