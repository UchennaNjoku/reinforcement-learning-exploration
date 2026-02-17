# Game 2: MiniGrid Wall Navigation 🧱

> **Custom Environment with Dynamic Obstacles**  
> A challenging navigation task requiring the agent to learn detour behavior around wall obstacles.

## Environment Description

**RandomGoalObstacleEnv** is a custom MiniGrid environment where:
- **10×10 playable grid** (12×12 with border walls)
- **Random agent start position** each episode
- **Random goal position** each episode  
- **Dynamic wall obstacle** placed between agent and goal, blocking the direct path
- **The agent must navigate around** the wall to reach the goal

### The Challenge
Unlike Frozen Lake where holes are static, the wall obstacle here:
- Appears at a random position between agent and goal (30%-70% of the path)
- Is **perpendicular** to the agent-goal line
- Forces the agent to learn **detour behavior** (go around, not through)

### Actions (3 discrete actions)
| Action | Value | Description |
|--------|-------|-------------|
| turn_left  | 0 | Rotate 90° counter-clockwise |
| turn_right | 1 | Rotate 90° clockwise |
| move_forward | 2 | Move one tile forward |

### State Representation Evolution
The state representation evolved significantly across agent versions:

**Early versions (v1-v5)**: Simple position-based states  
**Middle versions (v6-v10)**: Distance-based features  
**Current versions (v11-v13)**: Hybrid approach combining:
- Relative goal position (precise, ±4)
- Relative wall position (coarse, ±1)
- Tri-directional wall sensors (front/left/right blocked)

### Reward Structure
The reward shaping evolved through iterations:
- **Potential-based shaping**: Reward based on progress toward goal
- **Step penalty**: -0.01 per step (encourages efficiency)
- **Turn penalty**: -0.01 per turn (discourages spinning)
- **Wall bump penalty**: -0.05 for collision attempts
- **Timeout penalty**: -1.0 for hitting max steps

---

## Agent Iteration History 📈

This section documents the evolution of agents, showing what worked and what didn't.

### Summary Table

| Version | Key Innovation | Result | Status |
|---------|---------------|--------|--------|
| v1 | Basic position-based Q-learning | ~15% success | ❌ Failed |
| v2 | Distance to goal added | Wandering behavior | ❌ Failed |
| v3 | Distance to obstacle added | Slight improvement | ❌ Failed |
| v4 | Refined distance bins | Limited progress | ❌ Failed |
| v5 | Different coordinate system | No convergence | ❌ Failed |
| v6 | Simplified state space | ~20% success | ❌ Failed |
| v7 | Relative goal vector | ~25% success | ❌ Failed |
| v8 | Wall vector added | ~30% success | ⚠️ Better |
| v9 | Wall vector + front sensor | ~36% success | ⚠️ Promising |
| v10 | Tri-directional sensors only | ~23% success | ❌ Regressed |
| **v11** | **Hybrid: wall vector + tri-sensors** | **~70% success** | ✅ **BEST** |
| v12 | Curriculum learning support | Similar to v11 | ✅ Good |
| v13 | Visited-state penalty | ~65% success | ✅ Good |

### Detailed Agent Evolution

#### ❌ v1-v5: The Exploration Phase

**CustomAgent (v1) - Position + Distance**
- State: `(agent_x, agent_y, goal_x, goal_y, dist_goal, dist_obs)`
- Problem: State space too large (100×100×15×10 = 1.5M states)
- Result: Poor generalization, ~15% success

**CustomAgent2-5 - Various Attempts**
- Tried different distance precision levels
- Attempted coarse binning vs fine binning
- Added wall proximity features
- Result: None broke past 25% success rate

**Key Lesson**: Absolute positions don't generalize. The agent needs **relative** information.

---

#### ⚠️ v6-v8: The Relative Direction Era

**CustomAgent6-7 - Relative Goal Vector**
- State: `(dir, dxg, dyg)` where dxg/dyg = relative goal position
- Problem: No wall awareness - agent walks into walls
- Result: ~25% success (when goal is in clear sight)

**CustomAgent8 - Added Wall Vector**
- State: `(dir, dxg, dyg, dxw, dyw)`
- dxw/dyw = relative wall position (coarse)
- Result: ~30% success - agent can now "see" the wall

**Key Lesson**: Need both goal direction AND wall awareness, plus local sensors.

---

#### ⚠️ v9-v10: The Sensor Debate

**CustomAgent9 - Wall Vector + Front Sensor**
- State: `(dir, dxg, dyg, dxw, dyw, front_blocked)`
- Result: ~36% success
- Problem: Agent knows where wall is AND if front is blocked, but doesn't know which way to turn

**CustomAgent10 - Tri-Directional Sensors Only**
- State: `(dir, dxg, dyg, front_blocked, left_blocked, right_blocked)`
- Removed wall vector - only local sensors
- Result: ~23% success
- Problem: Agent avoids walls locally but forgets where the wall is globally

**Key Lesson**: You need **BOTH** wall position (for strategy) AND local sensors (for immediate safety).

---

#### ✅ v11: The Breakthrough

**CustomAgent11 - Hybrid Approach**

```python
State: (dir, dxg, dyg, dxw, dyw, front_blocked, left_blocked, right_blocked)
```

| Component | Purpose |
|-----------|---------|
| `dir` | Agent's facing direction (4 values) |
| `dxg, dyg` (±4) | Precise goal direction - for navigation |
| `dxw, dyw` (±1) | Coarse wall direction - for strategy |
| `front/left/right_blocked` | Immediate obstacle detection - for safety |

**State Space**: 4 × 9 × 9 × 3 × 3 × 2 × 2 × 2 = **~23,328 states**

**Why it works**:
1. **Precise goal vector** (9×9 grid): Agent knows exactly which way to go
2. **Coarse wall vector** (3×3 grid): Agent knows roughly where wall is to plan detour
3. **Tri-sensors**: Agent won't walk into walls when executing the detour

**Result**: **~70% success rate** with curriculum learning!

**Key Innovation**: The right balance of precision - fine for goals, coarse for walls.

---

#### ✅ v12-v13: Refinements

**CustomAgent12 - Curriculum Learning Support**
- Same state as v11
- Added explicit support for curriculum training
- No major changes to core logic

**CustomAgent13 - Visited-State Penalty (Intrinsic Motivation)**
- Same state representation as v11
- **New**: Tracks visited positions within each episode
- **New**: Penalizes revisiting positions (`-0.1` penalty)
- **New**: Action masking (can't move forward into walls)
- **New**: Tie-breaking preference (forward > left > right)

**Result**: ~65% success, but more consistent paths (less looping)

**Key Lesson**: Loop detection helps, but v11's state representation was the real breakthrough.

---

## Training & Usage

### Installation
```bash
pip install -r requirements.txt
```

### Training

#### Basic Training (recommended)
```bash
# Train with the best agent (v11) - default
python main.py train

# Train with more episodes
python main.py train --episodes 15000

# Train with curriculum learning (progressive difficulty)
python main.py train --agent CustomAgent13 --episodes 20000 --curriculum
```

#### Available Agents
```bash
python main.py train --agent CustomAgent     # Original distance-based
python main.py train --agent CustomAgent11   # Best overall (default)
python main.py train --agent CustomAgent13   # With loop detection
```

#### Advanced Options
```bash
# Custom epsilon decay parameters
python main.py train --epsilon-delay 1000 --epsilon-decay 0.999

# Render training (slow but educational)
python main.py train --render
```

### Evaluation
```bash
# Evaluate best checkpoint
python main.py eval

# Evaluate specific checkpoint
python main.py eval --checkpoint checkpoints/q_table_ep10000.pkl

# Evaluate without rendering (faster)
python main.py eval --no-render --episodes 1000
```

### Manual Play
```bash
python main.py manual
```
Controls: Arrow keys to move/turn, R to reset, Esc to quit

---

## What I Learned

### 1. State Representation is Everything
- Absolute positions → Don't generalize
- Pure distances → Not enough information
- **Relative vectors + local sensors** → Winner!

### 2. The Right Level of Abstraction
- Too fine (exact coordinates) → State space explosion
- Too coarse (just "near/far") → Not enough information
- **Fine for goals, coarse for obstacles** → Sweet spot

### 3. Exploration Strategy Matters
- Simple ε-greedy with decay → Okay
- **ε-delay (no decay for first N episodes)** → Better
- **Curriculum learning** → Best (learn goal chasing before obstacles)

### 4. Reward Shaping is Tricky
- Dense distance rewards → Can create local optima
- Potential-based shaping → Better, but still subtle
- **Simple penalties + let Q-learning figure it out** → Surprisingly effective

---

## Checkpoints

Saved checkpoints are in `checkpoints/`:
- `q_table.pkl` - Latest/best model
- `q_table_ep{N}.pkl` - Intermediate checkpoints every 1000 episodes

To load a specific checkpoint:
```bash
python main.py eval --checkpoint checkpoints/q_table_ep15000.pkl
```

---

## Next Steps

This environment taught me:
- Custom environment design with Gymnasium/MiniGrid
- State space engineering for discrete problems
- The importance of balancing exploration and exploitation

Ready for **Game 3** - likely a more complex environment with:
- Multiple obstacles
- Continuous state space (neural networks)
- Or perhaps multi-agent scenarios

See the main project README for the overall journey!
