# Game 4: CartPole DQN 🎯

> **The Jump to Deep RL**  
> Moving from Q-tables to neural networks — continuous states, experience replay, and a lot of debugging.

## Environment Description

**CartPole-v1** from Gymnasium. A pole is attached to a cart on a frictionless track. The agent pushes the cart left or right to keep the pole balanced.

### State Space (4 continuous values)
- Cart position
- Cart velocity
- Pole angle
- Pole angular velocity

This is why we can't use a Q-table anymore — these are continuous values, not discrete grid positions. A neural network learns to map these 4 numbers directly to Q-values.

### Actions
| Action | Value |
|--------|-------|
| Push left | 0 |
| Push right | 1 |

### Reward
- **+1** every timestep the pole stays up
- Episode ends when pole falls past ±12° or cart leaves bounds
- **Max return: 500** (episode truncates at 500 steps)

## Architecture

```
State (4) → Linear(128) → ReLU → Linear(128) → ReLU → Linear(2) → Q-values
```

Two identical networks:
- **Q-Network (online)** — selects actions, gets trained
- **Target Network** — provides stable targets for computing loss

## How to Run

```bash
cd games/cartpole-dqn

# Train
python main.py train --episodes 600 --lr 2.5e-4

# Watch trained agent
python main.py watch --episodes 5

# Play manually
python main.py manual --episodes 3
```

## What I Changed and Why

This was a lot of iterating. The vanilla DQN implementation worked but had serious instability problems — the agent would learn a great policy, then lose it completely within 50 episodes. Here's what I tried and what actually helped.

### Double DQN (DDQN)

Standard DQN uses the target network to both *pick* the best next action and *evaluate* it. This overestimates Q-values because you're always selecting the max.

DDQN fixes this by using the online network to select the action and the target network to evaluate it:

```python
# Online net picks the action
next_action = torch.argmax(q_network(s2), dim=1)
# Target net evaluates it
next_q = target_network(s2).gather(1, next_action.unsqueeze(1)).squeeze(1)
y = r + gamma * (~d).float() * next_q
```

This was the single biggest improvement. Without it the agent would peak around Avg50=90 and collapse. With it, the agent consistently hits 500-return episodes.

### Huber Loss instead of MSE

MSE squares the error, so outlier transitions with large TD errors dominate training. Huber loss acts like MSE for small errors but linear for large ones — basically gradient clipping built into the loss function.

```python
loss = nn.functional.smooth_l1_loss(q_val_chosen_action, y)
```

One line change, no downside.

### Gradient Clipping

Added after `loss.backward()` to prevent any single bad batch from blowing up the weights:

```python
torch.nn.utils.clip_grad_norm_(q_network.parameters(), 10.0)
```

### Polyak Soft Target Updates

Instead of copying the entire online network to the target network every N steps (which creates sudden jumps in targets), blend in 0.5% every step:

```python
tau = 0.005
with torch.no_grad():
    for target_p, online_p in zip(target_net.parameters(), q_net.parameters()):
        target_p.data.mul_(1 - tau)
        target_p.data.add_(tau * online_p.data)
```

The target network slowly tracks the online network. No more jarring target shifts.

### Evaluation-Based Early Stopping

This one took me a while to figure out. The training returns are noisy because of epsilon-greedy — the agent is randomly acting 1% of the time even late in training. I had runs where training Avg50 was only 28 but the actual greedy policy scored 466.

So I added a separate eval function that tests the pure greedy policy every 25 episodes:

```python
def evaluate_policy(q_net, env_name, episodes=10, device=None):
    # Runs 10 episodes with epsilon=0, returns mean and std
```

Training stops when eval hits 485+, and the best eval model gets checkpointed so we don't lose it if the policy collapses later (which it does — this is a known DQN thing).

### What Didn't Work

**Two gradient updates per environment step** — the idea was to extract more learning from the same experience. Instead it caused catastrophic forgetting. Long episodes (500 steps) meant 1000 gradient updates on highly correlated data, which rapidly overfitted the Q-network and crashed the policy.

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning Rate | 2.5e-4 | Lower = more stable, 1e-3 caused collapse |
| Batch Size | 64 | |
| Replay Buffer | 50,000 | |
| Warmup Steps | 2,000 | Fill buffer before training |
| Gamma | 0.99 | |
| ε Decay | 1.0 → 0.01 over 10k steps | 30k was way too slow |
| Target Update | Polyak τ=0.005 | Every training step |

## Key Differences from Games 1-3

| | Games 1-3 (Tabular) | Game 4 (DQN) |
|--|---------------------|--------------|
| State Space | Discrete | Continuous |
| Function Approximation | Q-Table | Neural Network |
| Memory | None | Experience Replay |
| Target Stability | N/A | Target Network + Polyak |

## Files

- `main.py` — Full implementation (train, watch, manual play)
- `cartpole_dqn.pt` — Final saved model
- `cartpole_best.pt` — Best model from eval checkpointing

## What's Next

- **Prioritized Experience Replay** — sample important transitions more often
- **Dueling DQN** — separate value and advantage streams
- **Harder environments** — LunarLander, Atari