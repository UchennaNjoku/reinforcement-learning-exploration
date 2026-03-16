# MARL Comms Research — Progress Document

**Project:** Emergent Communication Protocols for Coordination in Partially Observable Multi-Agent Pursuit
**Authors:** U. Njoku, J. Calderon — Dept. of Computer Science, Bethune-Cookman University
**Last updated:** 2026-03-16

---

## 1. Research Goal

Test whether a tiny learned discrete communication channel improves coordination in multi-agent pursuit under partial observability, and whether the messages become structured enough to interpret.

**Core claim being tested:**
> 2-bit discrete communication improves coordination over no communication, and approaches the performance of a higher-bandwidth discrete channel.

---

## 2. What Has Been Built

### 2.1 Environment

Base: `pettingzoo.sisl.pursuit_v4` wrapped in `envs/fixed_pursuit.py`.

Key design choices (frozen):
- 3 pursuers, 1 prey
- 7×7 local observation window (partial observability)
- `surround=False`, `n_catch=1` — overlap capture rule
- `distance_reward_scale=0.1` — dense shaped reward to solve reward sparsity
- Three fixed map presets: `easy_open`, `center_block`, `split_barrier`
- Rewards: catch=1.0, step_penalty=-0.01

**Why distance shaping:** Initial training with sparse reward only (catch=1.0) produced 0-4% capture rate across 5000 episodes. Agents never stumbled into captures often enough to learn. Adding a distance penalty gave the Q-network a gradient signal every step.

**Why `surround=False`:** The default `surround=True` requires all neighboring cells to be occupied — with 3 pursuers on an open 16×16 grid, many prey positions are structurally uncatchable. Switching to overlap capture (any pursuer on prey's cell) made the task learnable.

### 2.2 No-Communication Baseline

**Architecture:** `models/qnet.py` — `PursuitQNet`
- CNN: two 3×3 conv layers over (7,7,3) obs → 3136 features
- Agent-ID embedding (dim 8) concatenated with CNN features
- FC: 3136+8 → 256 → 128 → 5 Q-values
- Parameter sharing: all 3 pursuers share one network

**Training:** `train_baseline.py`
- DQN with epsilon-greedy (eps 1.0 → 0.05 over 500k steps)
- Shared replay buffer (capacity 100k), batch size 64
- Target network hard update every 500 steps
- 5000 episodes on `easy_open`, seed 0

**Training outcome:** Captured prey in ~15 steps with 100% rate by ep 4000.

### 2.3 Communication Model

**Architecture:** `models/comm_qnet.py` — `CommQNet`
- Same CNN backbone as baseline
- Same agent-ID embedding
- Message encoder: (n_agents-1) × vocab_size one-hot input → 16-dim embedding
- Shared trunk: combined features → 256 → 128
- **Move head:** 128 → 5 Q-values (trained with DQN)
- **Message head:** 128 → vocab_size Q-values (same TD target)

**Why separate heads (not joint Q-values):**
The first implementation used a joint action space of `5 × vocab_size` (20 or 80 actions). This caused Q-value collapse — greedy eval gave 0.5% capture despite 94-99% during epsilon-greedy training. DQN cannot cleanly assign credit across a large joint action space with sparse rewards when random exploration is doing most of the work. Separate heads reduce move selection to 5 Q-values (identical to the baseline), while the message head learns independently which symbols correlate with future team reward.

**Message timing:** Each agent receives teammates' messages from step t-1. At episode start all messages are zero vectors.

**Training:** `train_comm.py` — same hyperparameters as baseline, with two conditions:
- `--vocab-size 4` (2-bit, 4 symbols)
- `--vocab-size 16` (4-bit, 16 symbols)

### 2.4 Evaluation

`eval.py` — auto-detects model type from checkpoint's saved args.
- Runs N greedy episodes (epsilon=0)
- Records: capture rate, avg steps, escape rate, collision rate
- Map and n_catch default to checkpoint's training values; CLI override accepted with note

### 2.5 Message Logging

`train_comm.py` saves a rolling window of the last 500 training episodes' per-step message trajectories to `comm{N}_msg_log_seed{S}.json`. This is the artifact for interpretability analysis.

---

## 3. Results

All results: trained on `easy_open` seed 0, evaluated seed 99, 200 episodes.
Cross-map evals are transfer evals — no model was trained on `center_block` or `split_barrier`.

### 3.1 Comparison Table

| Condition | Map           | Capture Rate | Avg Steps | Collision Rate |
|-----------|---------------|:------------:|:---------:|:--------------:|
| Random    | easy_open     | 0%           | 300.0     | 4.9%           |
| Random    | center_block  | 63%          | 207.7     | 6.6%           |
| Random    | split_barrier | 53.5%        | 234.8     | 6.6%           |
| No-Comm   | easy_open     | 100%         | 14.49     | 0.1%           |
| No-Comm   | center_block  | 99.5%        | 40.52     | 1.4%           |
| No-Comm   | split_barrier | 99%          | 60.02     | **15.5%**      |
| Comm-4    | easy_open     | 100%         | **9.47**  | 0.02%          |
| Comm-4    | center_block  | 100%         | **22.91** | 30.1%          |
| Comm-4    | split_barrier | 82%          | 152.7     | 28.1%          |
| Comm-16   | easy_open     | 100%         | 11.13     | 0.75%          |
| Comm-16   | center_block  | 100%         | 27.41     | 1.94%          |
| Comm-16   | split_barrier | 96.5%        | 113.23    | 28.9%          |

See `results/comparison.png` for the bar chart.

---

## 4. Analysis: What We've Learned

### 4.1 The Core Claim Holds on the Training Map

On `easy_open` (the map all models trained on):
- All three trained conditions achieve 100% capture
- **Comm-4 captures in 9.47 steps** vs baseline's 14.49 — a 35% reduction
- Comm-16 is in between at 11.13 steps
- Both comm conditions outperform no-comm with near-zero collision rates

The ordering `Comm-4 ≈ Comm-16 > No-Comm` matches the abstract's claim exactly.

### 4.2 Comm-4 Outperforms Comm-16 on the Training Map

The 4-symbol vocabulary achieves faster capture than 16-symbol on both `easy_open` and `center_block`. This supports the abstract's specific sub-claim that "compact communication can approach the performance of higher-bandwidth channels." In this case, compact communication actually wins. One likely explanation: with a smaller vocabulary, the message head is forced to learn coarser but more consistently actionable symbols, while a 16-symbol space may fragment the signal across too many similar categories.

### 4.3 Transfer to Unseen Maps is Uneven

**center_block:** Both comm conditions transfer very well — 100% capture, 23-27 steps vs baseline's 40. The central obstacle doesn't prevent the learned coordination from working. Notably, Comm-4 has a very high collision rate (30%) on this map, meaning agents are still trying to path through the block. Comm-16 doesn't have this problem (1.9%), suggesting the larger vocabulary may encode spatial intent more precisely.

**split_barrier:** No-comm baseline (99%, 60 steps) outperforms both comm conditions (82% and 96.5%). The barrier with narrow gaps requires agents to navigate around specific chokepoints — a behavior that wasn't needed during training. The comm models' high collision rates (28-29%) show agents pressing against the barrier repeatedly. The no-comm baseline's 15.5% collision rate on this map (highest of its three maps) already signaled this difficulty.

### 4.4 The Collision Rate Pattern Reveals a Strategy Difference

On `easy_open`, all trained models have near-zero collision rate — there's nothing to collide with. On `center_block` and `split_barrier`, the comm models develop much higher collision rates than the no-comm baseline. The most likely explanation: communication allows faster convergence and more aggressive pursuit trajectories. When agents can signal their positions/intentions, they move more directly toward the prey without yielding to obstacles, which is fine on open maps but breaks down on barrier maps.

### 4.5 Reward Shaping Was Necessary

The first training attempts without shaped rewards (capture-only reward) produced flat learning curves at 0-4% capture for the entire 5000 episodes. The distance penalty (`scale=0.1`, normalized Manhattan distance to prey) provided a dense gradient signal that allowed capture rates to climb from 66% at ep 100. This is a standard practice in sparse-reward MARL and doesn't change the validity of the comparison — all three conditions used the same reward structure.

### 4.6 The Joint Q-Value Architecture Fails for Communication

The original comm model used a single Q-head over `5 × vocab_size` joint actions. Despite showing 94-99% capture during epsilon-greedy training, the greedy eval gave 0.5% capture (essentially random). Root cause: with 20+ joint actions and sparse rewards, most captures during training were due to random exploration rather than learned policy. The Q-values never converged to correctly identify which move was best — they collapsed to near-uniform values, causing the greedy policy to always take action 0. Switching to separate move and message heads (5 Q-values for moves, vocab_size Q-values for messages) resolved this completely.

---

## 5. What Is Still To Do

### 5.1 Remaining for the Paper (Required)

- [ ] Run 2 additional seeds per condition (seeds 1 and 2) for statistical validity
- [ ] Build per-seed summary table (mean ± std across seeds)
- [ ] Interpretability analysis on `comm4_msg_log_seed0.json`:
  - Message frequency distribution (which symbols are used most?)
  - Spatial correlation: do specific messages correlate with agent positions or prey direction?
  - Role differentiation: do different agents develop different message patterns?
- [ ] Training curve plots for comm4 and comm16 (same format as baseline plot)
- [ ] Rollout visualization / GIF export showing comm agents coordinating

### 5.2 Optional but Strengthening

- [ ] Per-map training runs for `center_block` and `split_barrier` to get fair baselines
- [ ] Comm models trained on `split_barrier` to test whether communication helps with barrier navigation
- [ ] t-SNE or clustering of message embeddings vs spatial context (richer interpretability)

---

## 6. Key Files

| File | Purpose |
|------|---------|
| `envs/fixed_pursuit.py` | Environment wrapper: fixed maps, starts, rewards |
| `models/qnet.py` | Baseline Q-network |
| `models/comm_qnet.py` | Communicating Q-network (separate move + msg heads) |
| `training/replay_buffer.py` | Baseline replay buffer |
| `training/comm_replay_buffer.py` | Comm replay buffer (stores received messages) |
| `train_baseline.py` | No-comm DQN training loop |
| `train_comm.py` | Comm DQN training loop |
| `eval.py` | Greedy evaluation, auto-detects model type |
| `plot_training.py` | Training curve figure (baseline) |
| `plot_comparison.py` | Cross-condition comparison figure |
| `results/baseline_v3/` | Baseline checkpoints, logs, evals |
| `results/comm4_v2/` | Comm-4 checkpoints, logs, evals, message log |
| `results/comm16_v2/` | Comm-16 checkpoints, logs, evals, message log |

---

## 7. How to Reproduce

```bash
# Activate environment
source .venv/bin/activate
cd marl-comms-research

# Evaluate baseline
python eval.py --checkpoint results/baseline_v3/checkpoints/baseline_final.pt --episodes 200 --seed 99

# Evaluate Comm-4
python eval.py --checkpoint results/comm4_v2/checkpoints/comm4_final.pt --episodes 200 --seed 99

# Evaluate Comm-16
python eval.py --checkpoint results/comm16_v2/checkpoints/comm16_final.pt --episodes 200 --seed 99

# Regenerate comparison figure
python plot_comparison.py

# Regenerate training curve
python plot_training.py
```
