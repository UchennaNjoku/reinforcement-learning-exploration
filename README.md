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
    ├── minigrid-wall-nav/             # Game 2: Custom environment
    │   ├── README.md
    │   ├── main.py
    │   ├── train.py
    │   ├── evaluate.py
    │   ├── config.py
    │   ├── utils.py
    │   ├── agents/                    # 13 agent iterations!
    │   │   ├── custom_agent.py
    │   │   ├── custom_agent_11.py     # The breakthrough
    │   │   └── ...
    │   ├── envs/
    │   │   └── random_goal_obstacle_env.py
    │   └── checkpoints/
    ├── gridworld-nav/                 # Game 3: Cardinal direction navigation
    │   ├── README.md
    │   ├── main.py
    │   ├── config.py
    │   ├── envs/
    │   │   └── grid_world_env.py
    │   ├── play_manual.py
    │   └── test_env.py
    └── cartpole-dqn/                  # Game 4: Deep Q-Networks
        ├── README.md
        ├── main.py
        ├── cartpole_dqn.pt
        └── cartpole_best.pt
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

### Game 3: GridWorld Navigation 🧭
**Simplifying Movement** | [Go to folder](./games/gridworld-nav/)

A custom Gymnasium environment where the agent navigates using cardinal directions (up/down/left/right) without orientation. Similar to Game 2 but with a simpler movement model.

**Key Differences from Game 2:**
| Feature | Game 2 (MiniGrid) | Game 3 (GridWorld) |
|---------|------------------|-------------------|
| Movement | Orientation-based (turn + forward) | Cardinal directions |
| Actions | 3 (turn left, turn right, forward) | 4 (up, down, left, right) |
| State | Position + Direction | Position only |
| Rendering | MiniGrid viewer | pygame (custom) |

**Key Learnings:**
- Custom Gymnasium environment creation
- BFS-based reachability checking
- pygame-based rendering
- Simpler action spaces can aid learning

---

### Game 4: CartPole with DQN 🎯
**The Leap to Deep RL** | [Go to folder](./games/cartpole-dqn/)

Deep Q-Network implementation for CartPole-v1. First foray into neural network-based function approximation for continuous state spaces.

**Key Learnings:**
- Neural networks as function approximators
- Experience replay for sample efficiency
- Target networks for stable training
- **Double DQN** to fix overestimation bias
- **Polyak averaging** for smooth target updates
- **Huber loss** for robust training
- **Gradient clipping** to prevent explosions
- **Evaluation-based stopping** (training returns are noisy!)

**Success Rate:** Solved (≥ 485 avg over 10 eval episodes)

**The Improvements:**
```python
# DDQN: Online selects, target evaluates
next_action = torch.argmax(q_network(s2), dim=1)
next_q = target_network(s2).gather(1, next_action.unsqueeze(1))

# Polyak soft update (τ=0.005)
target_p.data.mul_(1 - tau).add_(tau * online_p.data)
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

**Game 3 - GridWorld Navigation:**
```bash
cd games/gridworld-nav

# Play manually
python play_manual.py

# Test the environment
python test_env.py
```

**Game 4 - CartPole DQN:**
```bash
cd games/cartpole-dqn

# Train
python main.py train --episodes 600 --lr 2.5e-4

# Watch trained agent
python main.py watch --episodes 5

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
Adding Game 5 is simple:
```bash
mkdir games/game-5-name
# Add your README.md, train.py, etc.
# Update this main README
```

## Common Dependencies

```
gymnasium>=0.29.0      # Core RL environment API
minigrid>=2.3.0        # Gridworld environments  
numpy>=1.24.0          # Numerical operations
matplotlib>=3.7.0      # Training visualizations
torch>=2.0.0           # Neural networks (Game 4+)
pygame>=2.5.0          # Custom rendering (Game 3+)
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
| 5 | ??? (To Be Decided) | 🔜 Coming Soon |

Potential ideas:
- Policy Gradient methods (REINFORCE, Actor-Critic)
- Continuous action spaces (Pendulum, LunarLander)
- Multi-agent scenario
- Atari game (full DQN with CNN)
- Model-based RL

## Lessons Summary

| Concept | Game 1 | Game 2 | Game 3 | Game 4 |
|---------|--------|--------|--------|--------|
| **Environment** | Fixed grid | Dynamic obstacles | Custom cardinal nav | Continuous control |
| **State Space** | 16 discrete | ~23k discrete | Continuous positions | 4 continuous |
| **Algorithm** | Q-learning | Q-learning | (Environment only) | DQN / DDQN |
| **Key Challenge** | Credit assignment | State representation | Env design | Training stability |
| **Breakthrough** | ε-decay works | Hybrid state features | BFS reachability | DDQN + Polyak |
| **Winning Insight** | Simple is effective | Precision where it matters | Simple actions help | Separate eval from training |

## Acknowledgments

- **OpenAI Gymnasium** for the environment API
- **MiniGrid** for the customizable gridworld framework
- **PyTorch** for deep learning tools
- Classic RL textbooks (Sutton & Barto) for the fundamentals

---

> "The journey of a thousand miles begins with a single state."  
> — Ancient RL Proverb (probably)

**Happy Exploring! 🚀**
