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

# DQN Hyperparameters
AGENT_CONFIG = {
    "n_actions": 4,                # Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
    "learning_rate": 1e-3,         # Adam learning rate
    "discount_factor": 0.99,       # Gamma: discount factor
    "epsilon": 1.0,                # Initial exploration rate
    "epsilon_min": 0.05,           # Minimum epsilon
    "batch_size": 256,             # Replay buffer sample size
    "replay_size": 100_000,        # Replay buffer capacity
    "target_update_steps": 1000,   # Steps between target network updates
    "grad_clip": 10.0,             # Max gradient norm
}

# Training Configuration
TRAIN_CONFIG = {
    "n_episodes": 5000,            # Total training episodes (single-phase)
    "eval_interval": 500,          # Evaluate every N episodes
    "eval_episodes": 25,           # Episodes per evaluation
    "save_interval": 1000,         # Save checkpoint every N episodes
    "log_interval": 50,            # Print stats every N episodes
    "warmup_steps": 5000,          # Random steps before training starts
    "eps_decay_episodes": 12_000,  # Episodes to linearly decay epsilon over
}

# Paths
PATHS = {
    "save_dir": "./checkpoints",
    "logs_dir": "./logs",
}