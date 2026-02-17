# Reinforcement Learning Exploration 🎮🤖

> **A personal journey through machine learning, beginning with reinforcement learning**

This repository documents my exploration of reinforcement learning, starting from classic tabular Q-learning and progressively tackling more complex environments. Each "game" represents a distinct learning phase with its own challenges, solutions, and insights.

## Project Philosophy

```
Learn by doing.
```

This isn't just a collection of RL implementations—it's a documented learning journey showing:
- ✅ What worked
- ❌ What didn't  
- 💡 Key insights from each iteration
- 📈 Progression from simple to complex

## Repository Structure

```
reinforcement-learning-exploration/
├── README.md                          # This file
├── requirements.txt                   # Shared dependencies
└── games/                             # Each game is self-contained
    ├── frozen-lake/                   # Game 1: Classic tabular RL
    │   ├── README.md
    │   ├── train.py
    │   └── checkpoints/
    └── minigrid-wall-nav/             # Game 2: Custom environment
        ├── README.md
        ├── main.py
        ├── train.py
        ├── evaluate.py
        ├── config.py
        ├── utils.py
        ├── agents/                    # 13 agent iterations!
        │   ├── custom_agent.py
        │   ├── custom_agent_11.py     # The breakthrough
        │   └── ...
        ├── envs/                      # Custom environment
        │   └── random_goal_obstacle_env.py
        └── checkpoints/
```

## The Journey So Far

### Game 1: Frozen Lake ❄️
**The Foundation** | [Go to folder](./games/frozen-lake/)

Classic tabular Q-learning on a 4×4 grid. The agent learns to navigate from start to goal without falling into holes.

**Key Learnings:**
- Q-learning fundamentals
- Exploration vs exploitation (ε-greedy)
- Delayed reward propagation
- State-action value tables

**Success Rate:** ~95-100% on 4×4 non-slippery

---

### Game 2: MiniGrid Wall Navigation 🧱
**The Challenge** | [Go to folder](./games/minigrid-wall-nav/)

Custom environment with dynamic wall obstacles. The agent must learn to navigate around randomly placed walls blocking the direct path to the goal.

**Key Learnings:**
- State representation engineering
- Relative vs absolute positioning
- Combining multiple feature types (vectors + sensors)
- Curriculum learning
- **13 iterations** to find the winning approach!

**Success Rate:** ~70% with CustomAgent11

**The Breakthrough:**
```python
# The winning state representation:
(dir, dxg, dyg, dxw, dyw, front_blocked, left_blocked, right_blocked)

# Precise goal direction + coarse wall direction + local sensors
# = ~23k states that actually generalize!
```

---

## Getting Started

### Prerequisites
```bash
# Python 3.8+
pip install -r requirements.txt
```

### Running Each Game

**Game 1 - Frozen Lake:**
```bash
cd games/frozen-lake
python train.py                    # Train and visualize
python train.py --episodes 5000    # Custom episodes
```

**Game 2 - MiniGrid Wall Nav:**
```bash
cd games/minigrid-wall-nav

# Train with the best agent
python main.py train

# Train with curriculum learning
python main.py train --agent CustomAgent13 --curriculum --episodes 20000

# Evaluate
python main.py eval

# Play manually
python main.py manual
```

## Design Principles

### 1. Self-Contained Games
Each game folder is **completely independent**:
- Has its own README with full documentation
- Includes all necessary code
- Own checkpoints/ and logs/ directories
- Can be understood and run in isolation

### 2. Documented Iterations
For complex environments (Game 2+), each README includes:
- Agent iteration history
- What worked and what didn't
- State representation evolution
- Performance comparisons

### 3. Extensible Structure
Adding Game 3 is simple:
```bash
mkdir games/game-3-name
# Add your README.md, train.py, etc.
# Update this main README
```

## Common Dependencies

```
gymnasium>=0.29.0      # Core RL environment API
minigrid>=2.3.0        # Gridworld environments  
numpy>=1.24.0          # Numerical operations
matplotlib>=3.7.0      # Training visualizations
```

Each game may have additional specific requirements listed in its own README.

## Training Tips

### For Beginners
1. Start with **Game 1** - understand Q-learning basics
2. Read the README thoroughly before running code
3. Experiment with hyperparameters

### For Extension
1. Copy the structure of an existing game
2. Replace environment with your own
3. Adapt the agent state representation
4. Document your iterations!

## Future Games

| Game | Concept | Status |
|------|---------|--------|
| 3 | ??? (To Be Decided) | 🔜 Coming Soon |

Potential ideas:
- Continuous state space (CartPole with function approximation)
- Multi-agent scenario
- Atari game (Deep Q-Networks)
- Custom 3D navigation

## Lessons Summary

| Concept | Game 1 | Game 2 |
|---------|--------|--------|
| **Environment** | Fixed grid | Dynamic obstacles |
| **State Space** | 16 discrete | ~23k discrete |
| **Key Challenge** | Credit assignment | State representation |
| **Breakthrough** | ε-decay works | Hybrid state features |
| **Winning Insight** | Simple is effective | Precision where it matters |

## Acknowledgments

- **OpenAI Gymnasium** for the environment API
- **MiniGrid** for the customizable gridworld framework
- Classic RL textbooks (Sutton & Barto) for the fundamentals

---

> "The journey of a thousand miles begins with a single state."  
> — Ancient RL Proverb (probably)

**Happy Exploring! 🚀**
