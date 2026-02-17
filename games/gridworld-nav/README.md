# Game 3: GridWorld Navigation (Cardinal Directions)

A custom Gymnasium GridWorld environment where the agent navigates using cardinal directions (up/down/left/right) without orientation.

## Overview

This environment is Game 3 in the reinforcement learning exploration series. It shares the same core mechanics as Game 2 (MiniGrid Wall Navigation) but uses a simpler movement model:

- **Game 2 (MiniGrid)**: Agent has orientation (direction facing) + turn left/right/forward actions
- **Game 3 (GridWorld)**: Agent has no orientation + moves directly in 4 cardinal directions

## Environment Features

- **Grid Size**: 12×12 total (10×10 playable area with 1-tile border walls)
- **Action Space**: `Discrete(4)` - UP (0), DOWN (1), LEFT (2), RIGHT (3)
- **Observation Space**: Dict with agent position, goal position, and obstacle positions
- **Random Placement**: Agent, goal, and wall obstacle are randomly placed each episode
- **Blocking Wall**: A wall is placed perpendicular to the agent-goal line, blocking the direct path
- **Reachability Guarantee**: BFS check ensures the goal is always reachable from the starting position

## Installation

From the project root:

```bash
pip install -r requirements.txt
```

Required packages:
- gymnasium
- numpy
- pygame (for rendering)

## Usage

### Manual Play

Play the environment manually using keyboard controls:

```bash
cd games/gridworld-nav

# Graphical mode (requires pygame)
python play_manual.py

# Text mode (no pygame required)
python play_manual.py --text

# Larger grid
python play_manual.py --size 16
```

**Controls:**
- `W` / `↑` = Move UP
- `S` / `↓` = Move DOWN  
- `A` / `←` = Move LEFT
- `D` / `→` = Move RIGHT
- `R` = Reset environment
- `ESC` / `Q` = Quit

### Testing

Run automated tests:

```bash
# Run all tests
python test_env.py

# Run with rendering
python test_env.py --render

# Test action mappings only
python test_env.py --action-test
```

### Using as a Gymnasium Environment

```python
from envs.grid_world_env import GridWorldEnv

# Create environment
env = GridWorldEnv(
    size=12,                    # Grid size including borders
    max_steps=120,              # Maximum steps per episode
    wall_length=4,              # Length of blocking wall
    render_mode="human",        # "human", "rgb_array", "ansi", or None
)

# Reset environment
obs, info = env.reset()
print(f"Agent at: {obs['agent_pos']}")
print(f"Goal at: {obs['goal_pos']}")

# Take actions
action = 3  # RIGHT
obs, reward, terminated, truncated, info = env.step(action)

# Close environment
env.close()
```

## Action Space

| Action | Value | Description |
|--------|-------|-------------|
| UP     | 0     | Move up (y - 1) |
| DOWN   | 1     | Move down (y + 1) |
| LEFT   | 2     | Move left (x - 1) |
| RIGHT  | 3     | Move right (x + 1) |

## Observation Space

```python
{
    "agent_pos": np.array([x, y], dtype=np.int32),  # Agent coordinates
    "goal_pos": np.array([x, y], dtype=np.int32),   # Goal coordinates
    "obstacle_positions": [                          # List of wall positions
        np.array([x1, y1], dtype=np.int32),
        np.array([x2, y2], dtype=np.int32),
        ...
    ]
}
```

## Reward Function

- `+1.0` for reaching the goal
- `0.0` otherwise

Note: For training, you may want to implement additional reward shaping (e.g., distance penalties).

## Project Structure

```
games/gridworld-nav/
├── envs/
│   ├── __init__.py           # Environment exports
│   └── grid_world_env.py     # Main environment implementation
├── config.py                 # Configuration settings
├── play_manual.py            # Manual play script
├── test_env.py               # Test suite
└── README.md                 # This file
```

## Key Differences from Game 2

| Feature | Game 2 (MiniGrid) | Game 3 (GridWorld) |
|---------|------------------|-------------------|
| Movement | Orientation-based (turn + forward) | Cardinal directions |
| Actions | 3 (turn left, turn right, forward) | 4 (up, down, left, right) |
| State | Position + Direction | Position only |
| Rendering | MiniGrid viewer | pygame (custom) |
| Dependencies | minigrid | gymnasium + pygame |

## Next Steps

Future implementations for Game 3:

1. **Tabular Q-Learning**: Implement a basic Q-learning agent
2. **Reward Shaping**: Add distance-based rewards for faster learning
3. **Curriculum Learning**: Progressive difficulty (longer walls, smaller gaps)
4. **Visualization**: Training progress plots and heatmaps

## License

Part of the Reinforcement Learning Exploration project.
