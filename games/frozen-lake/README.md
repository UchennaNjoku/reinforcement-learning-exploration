# Game 1: Frozen Lake ❄️

> **First Steps in Reinforcement Learning**  
> A classic discrete environment for learning tabular Q-learning.

## Environment Description

**FrozenLake-v1** from OpenAI Gymnasium is a gridworld where an agent navigates a frozen lake to reach a goal without falling through holes.

### The Grid
- **4×4 or 8×8 grid** with:
  - **S** = Starting position (top-left)
  - **F** = Frozen surface (safe to walk)
  - **H** = Hole (fall in = episode ends, reward = 0)
  - **G** = Goal (reach = episode ends, reward = +1)

### Actions
| Action | Value | Description |
|--------|-------|-------------|
| LEFT   | 0     | Move left   |
| DOWN   | 1     | Move down   |
| RIGHT  | 2     | Move right  |
| UP     | 3     | Move up     |

### State Space
- Discrete: `0` to `N-1` (where N = 16 for 4×4, 64 for 8×8)
- Each state represents the agent's current position in the grid

### Reward Structure
- **+1** for reaching the goal
- **0** for all other transitions (including falling in holes)
- **No negative rewards** - the agent must learn from failure

## Learning Approach

### Tabular Q-Learning
This implementation uses classic Q-learning with a table storing Q-values for each (state, action) pair.

**Update Rule:**
```
Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
```

Where:
- `α` (alpha) = Learning rate (0.1)
- `γ` (gamma) = Discount factor (0.99)
- `ε` (epsilon) = Exploration rate (decays from 1.0 to 0.1)

### Key Features
- **Tie-breaking**: When multiple actions have the same Q-value, we randomly select among them for better exploration
- **Deterministic mode**: `is_slippery=False` for easier learning (agent moves exactly as commanded)
- **Episode-based training**: 10,000 episodes with ε-decay

## Training & Usage

### Basic Training
```bash
# Train with default settings (4x4 map, 10k episodes)
python train.py

# Train on 8x8 map
python train.py --map 8x8

# Train with slippery ice (harder!)
python train.py --slippery

# More episodes
python train.py --episodes 20000
```

### Evaluation
```bash
# Evaluate a saved Q-table
python train.py --evaluate checkpoints/q_table.npy
```

### Output Files
- `checkpoints/q_table.npy` - Saved Q-table after training

## What I Learned

This environment taught me the fundamentals:

1. **Exploration vs Exploitation**: The ε-greedy strategy helps the agent discover the goal before exploiting that knowledge
2. **Delayed Rewards**: With only +1 at the goal, the agent must propagate this reward backward through many states
3. **State Representation**: Simple discrete states work well for small gridworlds
4. **Convergence**: After ~5000 episodes, the agent reliably finds the optimal path

## Results

**4×4 Non-Slippery:**
- Converges in ~3000-5000 episodes
- Final success rate: ~95-100%
- Optimal path length: Typically 6 steps

## Next Steps

This simple environment prepared me for more complex challenges with:
- Continuous state spaces
- Custom environment design
- More sophisticated state representations

See [Game 2: MiniGrid Wall Navigation](../minigrid-wall-nav/) for the next iteration!
