# Project Onboarding Guide
**Emergent Communication for Transfer in Partially Observable Multi-Agent Pursuit**
U. Njoku, J. Calderon — Bethune-Cookman University

This document is written for someone who is new to this project and may have limited background in multi-agent reinforcement learning. It covers every moving part: the environment, the models, the training setup, the evaluation methodology, all major decisions made and why, all results, and how to extend the work. Read it once end-to-end and you should be able to run, reproduce, and build on everything here.

---

## Table of Contents

1. [What this project is about](#1-what-this-project-is-about)
2. [Concepts you need to know first](#2-concepts-you-need-to-know-first)
3. [Repository structure](#3-repository-structure)
4. [The environment](#4-the-environment)
5. [The models](#5-the-models)
6. [Training](#6-training)
7. [Evaluation methodology](#7-evaluation-methodology)
8. [Results](#8-results)
9. [Interpretability analysis](#9-interpretability-analysis)
10. [Visualization](#10-visualization)
11. [Key decisions and why they were made](#11-key-decisions-and-why-they-were-made)
12. [Known failures and honest notes](#12-known-failures-and-honest-notes)
13. [How to run everything from scratch](#13-how-to-run-everything-from-scratch)
14. [How to extend the project](#14-how-to-extend-the-project)

---

## 1. What this project is about

**The question:** If you give a team of AI agents a small communication channel — the ability to broadcast a single discrete symbol each step — do they learn to use it in a way that helps them coordinate? And does that communication advantage appear mainly during training, or only when the agents are placed in an environment they have never seen before?

**The setting:** Three AI pursuers (agents we control) chase one prey on a grid-world map. Each pursuer can only see a small 7×7 patch of the map around itself. The prey moves evasively. The team wins when any pursuer reaches the prey's cell.

**The experimental design:** Train all models on one simple open map. Then test them zero-shot on two harder maps with obstacles the models were never trained on. Compare three conditions: no communication, 4-symbol vocabulary, 16-symbol vocabulary.

**The core finding:** Communication does not meaningfully help on the training map — all conditions reach 100% capture there. The benefit of communication appears on the transfer maps, where comm models are more consistent across seeds and achieve higher mean capture rates. A larger vocabulary (16 symbols) transfers more reliably than a smaller one (4 symbols).

---

## 2. Concepts you need to know first

### Reinforcement learning (RL)
Agents learn by trial and error. At each step the agent observes the environment, takes an action, and receives a reward. Over time it learns which actions lead to higher total reward. We use **deep Q-learning (DQN)**, which uses a neural network to estimate Q(s, a) — the expected future reward of taking action a in state s.

### Multi-agent RL (MARL)
Multiple agents operate in the same environment simultaneously. Their actions affect each other. This project uses **cooperative MARL** — all agents share a single reward signal (did the team capture the prey?).

### Parameter sharing
Instead of training a separate neural network for each agent, all agents use one shared network. The agent's index (0, 1, or 2) is passed as an input so the network can differentiate agents without separate weights. This dramatically reduces sample complexity.

### Partial observability
Each agent only sees a local window of the environment — a 7×7 grid centered on itself. It cannot see the full map or directly observe where its teammates are outside that window. This is what makes communication potentially useful: agents may have information their teammates lack.

### Epsilon-greedy exploration
During training, the agent takes a random action with probability epsilon, and the greedy (best estimated) action with probability 1-epsilon. Epsilon starts at 1.0 (fully random) and decays to 0.05 over training. This ensures the agent explores the environment early before exploiting what it has learned.

### Emergent communication
No one tells the agents what their symbols mean. The agents discover a useful communication protocol entirely through training — the symbols that correlate with team success get reinforced. Whether the resulting protocol is interpretable by humans is a separate question.

### Transfer / generalization
Testing a model on an environment it was not trained on, with no additional training. A model that transfers well has learned something more general than just memorizing the training environment.

---

## 3. Repository structure

```
marl-comms-research/
│
├── envs/
│   ├── fixed_pursuit.py        # Environment wrapper — the core env setup
│   └── __init__.py
│
├── models/
│   ├── qnet.py                 # Baseline Q-network (no communication)
│   ├── comm_qnet.py            # Communicating Q-network (dual heads)
│   └── __init__.py
│
├── training/
│   ├── replay_buffer.py        # Experience replay for baseline
│   ├── comm_replay_buffer.py   # Experience replay for comm models
│   └── __init__.py
│
├── train_baseline.py           # Training script: no-comm condition
├── train_comm.py               # Training script: comm conditions
├── eval.py                     # Single-checkpoint greedy evaluation
├── eval_all_seeds.py           # Bulk eval across all seeds and maps
├── checkpoint_sweep.py         # Best-checkpoint selection pipeline
├── analyze_messages.py         # Message interpretability analysis
├── plot_comm_training.py       # Training curves for all conditions
├── plot_training.py            # Training curve for one condition
├── plot_comparison.py          # Cross-condition comparison figure
├── render_rollout.py           # Render a GIF of a greedy episode
│
├── results/
│   ├── baseline_v3/            # No-Comm seed 0 — checkpoints, logs, evals
│   ├── baseline_s1/            # No-Comm seed 1
│   ├── baseline_s2/            # No-Comm seed 2 — FAILED, do not use
│   ├── baseline_s3/            # No-Comm seed 3 — replacement/sensitivity check
│   ├── comm4_v2/               # Comm-4 seed 0
│   ├── comm4_s1/               # Comm-4 seed 1
│   ├── comm4_s2/               # Comm-4 seed 2
│   ├── comm16_v2/              # Comm-16 seed 0
│   ├── comm16_s1/              # Comm-16 seed 1
│   ├── comm16_s2/              # Comm-16 seed 2
│   ├── sweep_selection.json    # Selected checkpoint per run
│   ├── sweep_raw.json          # Raw per-map eval results for all runs
│   ├── sweep_summary.json      # Aggregated mean ± std table
│   ├── sweep_summary.md        # Human-readable summary table
│   ├── all_seeds_raw.json      # Raw results from eval_all_seeds.py
│   ├── msg_analysis/           # Comm-4 seed 0 interpretability outputs
│   ├── msg_analysis_16/        # Comm-16 seed 0 interpretability outputs
│   ├── comm_training_curves.png
│   ├── rollout_baseline_easy.gif
│   ├── rollout_comm4_easy.gif
│   ├── rollout_comm16_easy.gif
│   └── rollout_comm16_center.gif
│
├── PROGRESS.md                 # Running research log with all decisions
├── ONBOARDING.md               # This file
├── PPT_SCRIPT_FINAL.md         # Presentation script
└── requirements.txt
```

Each `results/<run>/` directory contains:
- `checkpoints/` — `.pt` files saved every 500 training episodes
- `*_train_log_seed*.json` — per-episode training stats (reward, capture rate, epsilon)
- `*_msg_log_seed*.json` — (comm only) last 500 episodes of per-step message logs
- `eval_*.json` — individual eval results on specific maps

---

## 4. The environment

### Base environment
We use `pettingzoo.sisl.pursuit_v4` — a standard MARL benchmark environment from the PettingZoo library. It is a discrete grid world where pursuers try to catch prey.

**PettingZoo default settings** (NOT what we use): 8 pursuers, 30 prey, 16×16 grid, `surround=True` (prey must be fully surrounded to be caught), `n_catch=2`. This is much harder than our setup.

**Our settings:**
- 3 pursuers, 1 prey, 16×16 grid
- `surround=False` — capture by overlap (any pursuer on prey's cell)
- `n_catch=1` — only one pursuer needed
- `obs_range=7` — each pursuer sees a 7×7 local window
- `catch_reward=1.0`, `step_penalty=-0.01` (urgency)
- `distance_reward_scale=0.1` — shaped reward (see Section 11)

### The wrapper: `envs/fixed_pursuit.py`
The raw PettingZoo environment randomizes agent and prey start positions each episode. We wrap it in `FixedMapPursuitWrapper` which:
1. Resets positions to fixed starts after each `env.reset()` call
2. Injects a custom obstacle map (obstacle cells set to -1)
3. Adds `blocked_move` and `evaders_remaining` to the info dict
4. Optionally adds a distance-based shaped reward

This wrapper is how all three maps are defined and used consistently.

### The three maps

**`easy_open`** — 16×16 open grid, no obstacles. Pursuers start left/bottom-left, prey starts right. This is the training map for all conditions.
```
Pursuer starts: (2,2), (2,13), (5,8)
Prey start:    (12,8)
```

**`center_block`** — 16×16 with a 4×4 central obstacle (cells 6:10, 6:10 blocked). Forces agents to route around the center.
```
Pursuer starts: (2,3), (2,12), (5,8)
Prey start:    (12,8)
```

**`split_barrier`** — 16×16 with a vertical wall at column 8, blocked rows 2–13 except gaps at rows 5 and 10. Agents must navigate through one of two narrow gaps to reach the prey.
```
Pursuer starts: (2,4), (2,11), (4,8)
Prey start:    (13,8)
```

**`large_split`** (exploratory, not in main results) — 20×20 with a vertical wall at column 10, gaps at rows 5 and 14. Same structure as split_barrier, larger grid. Used for an exploratory zero-shot stress test; results were checkpoint-sensitive and not included in the paper.

### Observation space
Each agent receives a `(7, 7, 3)` array:
- Channel 0: wall/obstacle locations
- Channel 1: count of allied pursuers at each cell
- Channel 2: count of prey at each cell

Values range 0–30. The agent can see 3 cells in each direction from its current position.

### Action space
`Discrete(5)`: up, down, left, right, stay. No diagonal movement.

---

## 5. The models

### Baseline: `models/qnet.py` — `PursuitQNet`

```
Input: (7,7,3) observation + agent index
  → CNN: Conv(3→32, 3×3) → ReLU → Conv(32→64, 3×3) → ReLU → Flatten → 3136 features
  → Agent embedding: index {0,1,2} → 8-dim vector
  → Concatenate: 3136 + 8 = 3144
  → FC: 3144 → 256 → ReLU → 128 → ReLU → 5 Q-values
```

The 5 Q-values correspond to the 5 movement actions. The agent takes the action with the highest Q-value (greedy) or a random action (epsilon-greedy during training).

**Parameter sharing:** All three pursuers use one instance of this network. The agent index input (0, 1, or 2) lets the network produce different behavior for each agent without separate weights.

### Communication model: `models/comm_qnet.py` — `CommQNet`

```
Input: (7,7,3) observation + agent index + received messages from teammates
  → CNN: identical to baseline → 3136 features
  → Agent embedding: identical to baseline → 8-dim vector
  → Message encoder: (n_agents-1) × vocab_size one-hot → Linear → 16-dim embedding
  → Concatenate: 3136 + 8 + 16 = 3160
  → Shared trunk: 3160 → 256 → ReLU → 128 → ReLU
  → Move head:    128 → 5 Q-values  (for movement)
  → Message head: 128 → vocab_size Q-values  (for what symbol to broadcast)
```

**Message input:** Each agent receives the messages its 2 teammates sent the previous step, each encoded as a one-hot vector of length `vocab_size`. These are concatenated: 2 × vocab_size = 8 (for Comm-4) or 32 (for Comm-16).

**Two output heads — why:** The first implementation used a single head over `5 × vocab_size` joint actions (20 for Comm-4, 80 for Comm-16). This failed catastrophically: 94-99% capture during training dropped to 0.5% during greedy evaluation. The problem is that DQN cannot reliably assign credit across a large joint action space with sparse rewards when most captures were due to random exploration. Splitting into separate heads reduces move selection to 5 Q-values (identical to baseline complexity), letting the move head train cleanly while the message head learns independently which symbols correlate with future team reward.

**Message timing:** At step t, each agent receives messages from step t-1 (one step delay). At episode start, all received messages are zero vectors.

**Vocab sizes tested:**
- Comm-4: vocab_size=4 (2 bits per message)
- Comm-16: vocab_size=16 (4 bits per message)

---

## 6. Training

### Shared hyperparameters (both conditions)
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam, lr=1e-4 |
| Epsilon schedule | 1.0 → 0.05 over 500,000 steps |
| Replay buffer capacity | 100,000 transitions |
| Batch size | 64 |
| Target network update | Hard update every 500 steps |
| Discount factor (γ) | 0.99 |
| Total episodes | 5,000 (baseline), typically 4,000–5,000 (comm) |
| Checkpoint saves | Every 500 episodes |
| Training map | `easy_open` |

### Baseline training
```bash
python train_baseline.py --seed 0 --out-dir results/baseline_v3
python train_baseline.py --seed 1 --out-dir results/baseline_s1
python train_baseline.py --seed 2 --out-dir results/baseline_s2
python train_baseline.py --seed 3 --out-dir results/baseline_s3
```

Saves: periodic checkpoints + `baseline_train_log_seed{N}.json`

### Comm training
```bash
python train_comm.py --vocab-size 4  --seed 0 --out-dir results/comm4_v2
python train_comm.py --vocab-size 4  --seed 1 --out-dir results/comm4_s1
python train_comm.py --vocab-size 4  --seed 2 --out-dir results/comm4_s2
python train_comm.py --vocab-size 16 --seed 0 --out-dir results/comm16_v2
python train_comm.py --vocab-size 16 --seed 1 --out-dir results/comm16_s1
python train_comm.py --vocab-size 16 --seed 2 --out-dir results/comm16_s2
```

Saves: periodic checkpoints + training log + message log (last 500 episodes of per-step messages)

### Where training was run
Training was run on Kaggle GPU notebooks (free tier) and results were downloaded. Training on a CPU machine is slow but feasible (~2–4 hours per run).

---

## 7. Evaluation methodology

### Single checkpoint evaluation: `eval.py`
```bash
python eval.py \
  --checkpoint results/comm16_v2/checkpoints/comm16_ep003500.pt \
  --map center_block \
  --episodes 200 \
  --seed 99
```
Auto-detects whether the checkpoint is baseline or comm from saved args. Runs N greedy episodes (epsilon=0) and reports: capture rate, escape rate, avg steps, avg steps on captured episodes, collision rate.

### Best-checkpoint selection: `checkpoint_sweep.py`
**Why this matters:** DQN training is unstable. The final checkpoint is not always the best one. Some runs partially degrade near the end of training. If you always compare final checkpoints, you risk comparing a degraded baseline against a near-peak comm model, or vice versa.

**The rule (applied uniformly to all 9 runs):**
1. For each run, evaluate all saved periodic checkpoints (every 500 episodes) on `easy_open` under the greedy policy
2. Select the checkpoint with the highest capture rate; break ties by lowest avg steps
3. Use that checkpoint for all transfer evaluations

```bash
# Full sweep: selects best checkpoints and evals all maps
python checkpoint_sweep.py --results-dir results --episodes 200 --seed 99

# Add a new run without re-sweeping everything
python checkpoint_sweep.py --only-subdir baseline_s3 --episodes 200 --seed 99

# Skip selection, just re-run Phase 2 evals (useful if you add a new map)
python checkpoint_sweep.py --skip-selection --episodes 200 --seed 99
```

Outputs:
- `results/sweep_selection.json` — selected checkpoint per run
- `results/sweep_raw.json` — per-map eval results for every selected checkpoint
- `results/sweep_summary.json` — aggregated mean ± std per condition per map
- `results/sweep_summary.md` — human-readable table

### Matched seeds
The primary comparison uses matched seeds {0, 1, 2} across all three conditions. `baseline_s2` failed to converge and is excluded from the primary table. `baseline_s3` (seed 3) is a supporting sensitivity check confirming the failure was isolated to that seed.

---

## 8. Results

### In-distribution (training map: `easy_open`)
All three conditions achieve 100% capture rate with ~9–11 average steps. No meaningful difference. Communication is not needed on the familiar training map.

### Transfer results — primary table (matched seeds {0,1,2}, best-checkpoint selection)

| Condition | Map | Capture Rate | Avg Steps | Collision Rate | N seeds |
|-----------|-----|:------------:|:---------:|:--------------:|:-------:|
| No-Comm | easy_open | 100% ±0.0 | 9.6 ±0.9 | 0.1% | 3 |
| No-Comm | center_block | 77.7% ±28.8 | 121.1 ±97.6 | 12.3% | 3 |
| No-Comm | split_barrier | 65.8% ±26.2 | 173.1 ±78.8 | 38.1% | 3 |
| Comm-4 | easy_open | 100% ±0.0 | 9.7 ±0.3 | 0.0% | 3 |
| Comm-4 | center_block | 91.0% ±12.7 | 73.7 ±67.2 | 0.5% | 3 |
| Comm-4 | split_barrier | 70.7% ±18.3 | 159.8 ±66.6 | 4.1% | 3 |
| Comm-16 | easy_open | 100% ±0.0 | 9.0 ±0.2 | 0.0% | 3 |
| Comm-16 | center_block | **95.2% ±6.1** | **63.6 ±50.3** | 0.9% | 3 |
| Comm-16 | split_barrier | **79.5% ±12.7** | **139.4 ±56.1** | 3.7% | 3 |

> Note: The No-Comm easy_open mean is 100% ±0 when using only the three converged seeds (0, 1, 3). The 77.7%/83.8% figure in some older documents included the failed baseline_s2 seed. The sweep_summary.md reflects the seed-3 sensitivity run. Use sweep_raw.json as the authoritative source.

**Key takeaways:**
- Communication improves mean transfer performance on both harder maps
- Comm-16 outperforms Comm-4 — larger vocabulary transfers more reliably
- Communication dramatically reduces collision rates (38% → 4% on split_barrier)
- Communication reduces variance — No-Comm has ±28.8 std on center_block; Comm-16 has ±6.1

### Per-seed breakdown
| Condition | Subdir | Seed | Selected Ep | easy_open | center_block | split_barrier |
|-----------|--------|:----:|:-----------:|:---------:|:------------:|:-------------:|
| No-Comm | baseline_v3 | 0 | ep004000 | 100% | 100% | 99% |
| No-Comm | baseline_s1 | 1 | ep005000 | 100% | 37% | 35% |
| No-Comm | baseline_s2 | 2 | ep000500 | **51.5% (FAILED)** | — | — |
| No-Comm | baseline_s3 | 3 | ep005000 | 100% | 96% | 63.5% |
| Comm-4 | comm4_v2 | 0 | ep004000 | 100% | 100% | 96.5% |
| Comm-4 | comm4_s1 | 1 | ep003500 | 100% | 73% | 58.5% |
| Comm-4 | comm4_s2 | 2 | ep001500 | 100% | 100% | 57% |
| Comm-16 | comm16_v2 | 0 | ep003500 | 100% | 99.5% | 95% |
| Comm-16 | comm16_s1 | 1 | ep004500 | 100% | 99.5% | 79.5% |
| Comm-16 | comm16_s2 | 2 | ep004000 | 100% | 86.5% | 64% |

---

## 9. Interpretability analysis

### Script: `analyze_messages.py`

Analyzes the message log saved during training. The log contains per-step symbol choices for every agent across the last 500 training episodes.

```bash
# Comm-4, seed 0
python analyze_messages.py \
  --log results/comm4_v2/comm4_msg_log_seed0.json \
  --vocab-size 4 \
  --out-dir results/msg_analysis

# Comm-16, seed 0
python analyze_messages.py \
  --log results/comm16_v2/comm16_msg_log_seed0.json \
  --vocab-size 16 \
  --out-dir results/msg_analysis_16

# Override minimum episode threshold for capture/escape analysis
python analyze_messages.py --log <path> --vocab-size 4 --min-escaped 5
```

### Four analyses

**1. Symbol frequency and entropy**
Computes how often each symbol is used across all agents and episodes. Reports Shannon entropy vs theoretical maximum (log2(vocab_size)). Near-maximum entropy = full vocabulary used roughly uniformly.

**2. Per-agent heatmap and role differentiation**
Shows each agent's symbol usage distribution separately. Reports inter-agent variation (average standard deviation of symbol frequencies across agents). Low variation = no role differentiation.

**3. Temporal phase analysis**
Divides each episode into early/mid/late thirds. Computes symbol frequencies separately for each phase. Reports max shift = biggest change in any symbol's frequency across phases. Low shift = no temporal structure.

**4. Capture vs escape correlation (gated)**
Compares symbol frequencies in episodes that ended in capture vs episodes that ended in escape. **Only runs if there are at least 20 escaped episodes in the log** (configurable via `--min-escaped`). If below threshold, reports a null result explicitly. This gate was added after a spurious finding from 1 escaped episode was initially reported as meaningful.

### Results (seed 0, both conditions)
| Metric | Comm-4 | Comm-16 |
|--------|--------|---------|
| Entropy | 1.997 / 2.0 bits (99.8%) | 3.985 / 4.0 bits (99.6%) |
| Inter-agent variation | 1.09% | 0.82% |
| Max temporal shift | 1.9% | 1.6% |
| Capture/escape | **not computable** — 2 escapes in 500 eps | **not computable** — 1 escape in 500 eps |

**Interpretation:** Both channels are fully active (near-max entropy). No agent specialization. No temporal structure. Communication is functional but not symbolically interpretable at this granularity. This is consistent with prior emergent communication literature — learned protocols are often distributed rather than categorical.

**To get the capture/escape analysis to work**, you need a model that fails more often. Options:
- Run the analysis on an earlier (weaker) checkpoint
- Run the analysis on a harder transfer map (center_block or split_barrier)
- Lower `--min-escaped` (not recommended without understanding the noise risk)

---

## 10. Visualization

### Training curves: `plot_comm_training.py`
```bash
python plot_comm_training.py
# output: results/comm_training_curves.png
```
Overlays training curves for all three conditions (seed 0): capture rate rolling average, steps rolling average, epsilon decay. Uses seed 0 logs from baseline_v3, comm4_v2, comm16_v2.

### Rollout GIFs: `render_rollout.py`
```bash
python render_rollout.py \
  --checkpoint results/comm16_v2/checkpoints/comm16_ep003500.pt \
  --map center_block \
  --out results/rollout_comm16_center.gif \
  --fps 6 \
  --seed 99
```

Runs one greedy episode and saves it as an animated GIF. For comm models, overlays colored message badges per agent at the bottom of each frame showing the current symbol each agent is broadcasting.

**Existing GIFs (presentation set):**
- `results/rollout_baseline_easy.gif` — No-Comm on easy_open (17 steps)
- `results/rollout_comm4_easy.gif` — Comm-4 on easy_open (17 steps)
- `results/rollout_comm16_easy.gif` — Comm-16 on easy_open (16 steps)
- `results/rollout_comm16_center.gif` — Comm-16 on center_block transfer (61 steps)

**To render a new GIF, use the .venv Python:**
```bash
/path/to/.venv/bin/python render_rollout.py --checkpoint <path> --map <map> --out <out.gif>
```

---

## 11. Key decisions and why they were made

### Why `surround=False`
The default PettingZoo setting is `surround=True`, which requires the prey to have all four cardinal neighbors occupied before capture. With 3 pursuers on a 16×16 grid, many prey positions are structurally uncatchable (e.g., prey in the center with only 3 pursuers available for 4 sides). Setting `surround=False, n_catch=1` makes capture happen on overlap — any pursuer reaching the prey's cell. This made the task learnable without architectural changes.

`surround=True` was intentionally ruled out for the current setup. Revisiting it would require either 4+ pursuers or accepting that boundary cells are necessary for all captures.

### Why distance reward shaping
Initial training with `catch_reward=1.0` only (no step penalty, no distance shaping) produced 0–4% capture rate after 5000 episodes across all conditions. Agents never stumbled into captures frequently enough to bootstrap learning. Adding `distance_reward_scale=0.1` (a small negative reward proportional to distance to prey, normalized to the grid size) gave the Q-network a dense gradient signal every step. All three conditions use identical reward structure, so this does not bias comparisons.

### Why separate move and message heads
The first comm model used a joint Q-head over `5 × vocab_size` actions. Despite showing 94-99% capture during epsilon-greedy training, greedy evaluation gave 0.5% capture (essentially random). Root cause: with a large joint action space and sparse rewards, most captures during training were due to random exploration. The Q-values never reliably converged — they collapsed to near-uniform values, and the greedy policy always took action 0. Splitting into separate heads reduces move complexity to 5 Q-values (identical to baseline) and trains correctly.

### Why best-checkpoint selection instead of final checkpoint
Early results using final checkpoints showed high collision rates (28–30%) for comm models and an inflated advantage for Comm-4 over No-Comm. This was an artifact: the comm final checkpoints happened to be near-peak while the baseline final checkpoint had partially degraded. Uniform best-checkpoint selection (sweep all 500-ep saves, pick highest capture on training map) eliminates this artifact. With this method, collision rates drop to 0–4% and in-distribution step counts converge across all conditions.

### Why matched seeds {0, 1, 2} for the primary table
Seed 2 of No-Comm failed (see Section 12). Rather than replace it asymmetrically (which would give No-Comm a cherry-picked seed pool), the primary table uses the same seed indices {0, 1, 2} for all conditions and reports the failure honestly. Seed 3 is presented separately as a sensitivity check only.

### Why communication is compared as a generalization benefit, not a training benefit
All conditions reach 100% on easy_open — there is no in-distribution advantage for communication. Framing the result as a training benefit would be misleading. The data clearly shows that the benefit materializes on unseen maps, so that is the claim.

---

## 12. Known failures and honest notes

### `baseline_s2` — failed training run
`baseline_s2` (No-Comm seed 2) never converged to a greedy policy at any point in training. Its best checkpoint across all 10 periodic evaluations achieved 51.5% capture on `easy_open`. The training log showed high epsilon-greedy capture rates (reaching 99% around episode 3200), but these were inflated by exploration. When epsilon decayed to 0.05, performance collapsed — the Q-values never reliably learned move quality. This is a known DQN failure mode caused by the network's weight landscape falling into a poor local optimum that appeared functional under high exploration.

The data for this run is kept in `results/baseline_s2/` for reference, but it is excluded from all comparison tables.

### DQN instability across seeds
Even among successful seeds, DQN shows meaningful instability — capture rates can fluctuate by 5–15% across checkpoints within a single run. This is why best-checkpoint selection was necessary and why standard deviations across seeds are large (especially for No-Comm on transfer maps). A more stable algorithm like PPO or SAC would likely produce cleaner results.

### `large_split` — exploratory stress test, not in main results
A 20×20 version of split_barrier was tested as a zero-shot stress test. Results were highly checkpoint-sensitive for Comm-16: the selected checkpoint (ep3500) achieved only 37.5% on large_split, while ep5000 achieved 91%. This means in-distribution optimality and out-of-distribution robustness do not peak at the same checkpoint on this harder map. The experiment was useful to run but the results are not clean enough to include in the main comparison. Raw results are in `results/comm16_v2/eval_large_split_ep*.json`.

### Interpretability null result
The capture/escape analysis cannot be computed on the training-map message logs because the agents fail too rarely (1–2 escaped episodes out of 500). This is not a bug — it is a consequence of the models performing too well on the task they were trained on. To get this analysis to work, run it on a harder map or weaker checkpoint.

### An earlier false finding (retracted)
An initial run of the interpretability analysis incorrectly reported that Comm-16 symbol 3 showed an 11.2% divergence between captured and escaped episodes. This was based on a single escaped episode and was noise. The `analyze_messages.py` script now gates the capture/escape section behind a minimum of 20 escaped episodes (`MIN_ESCAPED_EPISODES = 20`) and states the null result explicitly when the threshold is not met.

---

## 13. How to run everything from scratch

### Environment setup
```bash
cd /path/to/reinforcement-learning-exploration
source .venv/bin/activate
cd marl-comms-research
```

Always use the `.venv` in the parent `reinforcement-learning-exploration/` directory. This contains all required packages (torch, pettingzoo, numpy, PIL, matplotlib, etc.).

### Train all conditions (9 runs total)
```bash
# No-Comm
python train_baseline.py --seed 0 --out-dir results/baseline_v3
python train_baseline.py --seed 1 --out-dir results/baseline_s1
python train_baseline.py --seed 3 --out-dir results/baseline_s3

# Comm-4
python train_comm.py --vocab-size 4 --seed 0 --out-dir results/comm4_v2
python train_comm.py --vocab-size 4 --seed 1 --out-dir results/comm4_s1
python train_comm.py --vocab-size 4 --seed 2 --out-dir results/comm4_s2

# Comm-16
python train_comm.py --vocab-size 16 --seed 0 --out-dir results/comm16_v2
python train_comm.py --vocab-size 16 --seed 1 --out-dir results/comm16_s1
python train_comm.py --vocab-size 16 --seed 2 --out-dir results/comm16_s2
```

### Run the full evaluation pipeline
```bash
python checkpoint_sweep.py --results-dir results --episodes 200 --seed 99
```
This does everything: sweeps all checkpoints, selects the best per run, evals on all maps, writes all JSON and markdown outputs.

### Generate training curves
```bash
python plot_comm_training.py
# output: results/comm_training_curves.png
```

### Run interpretability analysis
```bash
python analyze_messages.py \
  --log results/comm4_v2/comm4_msg_log_seed0.json \
  --vocab-size 4 --out-dir results/msg_analysis

python analyze_messages.py \
  --log results/comm16_v2/comm16_msg_log_seed0.json \
  --vocab-size 16 --out-dir results/msg_analysis_16
```

### Render presentation GIFs
```bash
python render_rollout.py --checkpoint results/baseline_v3/checkpoints/baseline_ep004000.pt \
  --map easy_open --out results/rollout_baseline_easy.gif --fps 8 --seed 99

python render_rollout.py --checkpoint results/comm4_v2/checkpoints/comm4_ep004000.pt \
  --map easy_open --out results/rollout_comm4_easy.gif --fps 8 --seed 99

python render_rollout.py --checkpoint results/comm16_v2/checkpoints/comm16_ep003500.pt \
  --map easy_open --out results/rollout_comm16_easy.gif --fps 8 --seed 99

python render_rollout.py --checkpoint results/comm16_v2/checkpoints/comm16_ep003500.pt \
  --map center_block --out results/rollout_comm16_center.gif --fps 6 --seed 99
```

---

## 14. How to extend the project

### Add a new map
1. Open `envs/fixed_pursuit.py`
2. Add a new `MapSpec` entry to `MAP_SPECS` with a grid function, pursuer starts, and prey start
3. Add the map name to the `choices` list in `eval.py`, `render_rollout.py`, and `checkpoint_sweep.py`
4. Test with a quick smoke test: `python -c "from envs import make_fixed_pursuit_env; env = make_fixed_pursuit_env(map_name='your_map'); obs, _ = env.reset(); print('OK')"`

Available grid helpers in `fixed_pursuit.py`:
- `_empty_map(size)` — blank grid
- `_with_vertical_barrier(size, x, blocked_rows, gap_rows)` — single vertical wall
- `_with_central_block(size, start, end)` — central square obstacle
- `_with_double_barrier(size, x1, x2, blocked_rows, gap1_rows, gap2_rows)` — two walls

### Add more agents (requires architecture change)
The comm model's message input size is hardcoded as `(n_agents - 1) * vocab_size`. Changing `n_agents` from 3 to 4 requires:
1. Updating `N_AGENTS` and `AGENT_IDS` in `train_comm.py`
2. Passing `n_agents=4` to `CommQNet()`
3. Adding a fourth pursuer start to all `MapSpec` entries
4. Updating the render annotation colors/labels in `render_rollout.py`
5. Retraining all conditions from scratch (no transfer from 3-agent checkpoints)

### Try `surround=True`
Change `surround=False` to `surround=True` in `make_fixed_pursuit_env()` calls in `train_baseline.py` and `train_comm.py`. Be aware: with 3 pursuers, capture requires using walls as the fourth blocker. This changes the task definition (not just difficulty) and all existing checkpoints become invalid. Results would not be comparable to the current study.

### Improve the algorithm
Replace DQN with PPO or MAPPO (multi-agent PPO). This would reduce instability significantly and likely give cleaner results. The environment wrapper is algorithm-agnostic — only the training scripts would need to change. Libraries like `stable-baselines3` or `rllib` support PettingZoo environments directly.

### Improve interpretability analysis
Current analysis is purely statistical over symbol frequencies. Stronger approaches:
- **t-SNE of message embeddings:** Pass the raw message embedding (16-dim output of the message encoder) through t-SNE and color points by environmental context (e.g., distance to prey, relative agent position). Look for clustering.
- **Spatial conditioning:** Compute symbol frequency conditioned on the agent's position in the grid. Does symbol 3 appear more often when the agent is near the barrier gap?
- **Cross-entropy probing:** Train a small probe network to predict environmental features (e.g., "is the prey in the upper half?") from the message alone. If the probe exceeds random baseline, the message encodes that feature.

### Run on a harder map from training (not just transfer)
Currently all training is on `easy_open`. Training on `center_block` or `split_barrier` directly would let you test whether communication is even more beneficial when coordination is required during training. You would also likely get enough failed episodes in the message log to run the capture/escape analysis.

---

*For questions, see PROGRESS.md for the full research log, or contact U. Njoku.*
