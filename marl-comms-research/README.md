# MARL Comms Research

This folder contains a multi-agent reinforcement learning project built around one simple question:

**Do tiny learned messages help a team coordinate better than no communication at all?**

The task is a pursuit game with partial observability. Three pursuer agents try to catch one prey while only seeing a small local window of the map. We compare three settings:

- `No-Comm`: agents act without an explicit communication channel
- `Comm-4`: agents can send one of 4 symbols each step
- `Comm-16`: agents can send one of 16 symbols each step

The code here covers the full pipeline: environment setup, model definitions, training, evaluation, checkpoint sweeping, message analysis, plotting, and rollout visualization.

## Project In Plain English

The current results point to a clear high-level story:

- all three conditions can solve the simple training map
- communication helps more with **generalization to new maps** than with raw training-map performance
- the `16`-symbol channel is the strongest communication condition so far
- the learned messages are useful, but they do **not** look like neat human-readable symbols

So the safest takeaway is:

**communication seems most valuable here as a stability and transfer aid, not as a way to dramatically increase final training-map performance**

## What Has Been Built

The project already includes:

- a fixed-map partial-observability pursuit environment
- a no-communication DQN baseline
- communication-enabled models with separate move and message heads
- greedy evaluation on both training and transfer maps
- multi-seed checkpoint selection using a uniform sweep rule
- training-curve plots
- message interpretability analysis
- rollout GIFs for presentation/demo use

## Problem Setup

This project uses `pettingzoo.sisl.pursuit_v4` through a custom wrapper in `envs/fixed_pursuit.py`.

Frozen setup:

- `3` pursuers, `1` prey
- `7x7` local observation window
- overlap capture rule: `surround=False`, `n_catch=1`
- dense reward shaping with `distance_reward_scale=0.1`
- fixed map presets:
  - `easy_open`
  - `center_block`
  - `split_barrier`

Why these choices matter:

- without reward shaping, learning barely moved
- with the default surround-capture rule, many prey positions were structurally uncatchable
- fixed maps and fixed starts make condition-to-condition comparisons fair and reproducible

## Project Architecture

### Environment layer

- `envs/fixed_pursuit.py`
  - wraps PettingZoo pursuit
  - defines the fixed maps and reward/capture settings
  - exposes a stable environment interface for both training and evaluation

### Model layer

- `models/qnet.py`
  - baseline shared Q-network
  - CNN over the local observation window
  - agent-ID embedding so one shared policy can still condition on which pursuer is acting

- `models/comm_qnet.py`
  - communication model
  - uses the same observation backbone as the baseline
  - receives teammates' messages from the previous step
  - has two separate heads:
    - move head for action selection
    - message head for symbol selection

Why the two-head design matters:

- the first communication version used one big joint action space (`move x message`)
- it looked good during noisy epsilon-greedy training but collapsed under greedy evaluation
- separating movement from message prediction fixed that failure mode

### Training layer

- `train_baseline.py`
  - trains the no-communication DQN baseline

- `train_comm.py`
  - trains the communication models
  - mirrors the baseline training structure closely so comparisons stay clean
  - saves late-training message logs for analysis

- `training/replay_buffer.py`
  - replay buffer for the baseline

- `training/comm_replay_buffer.py`
  - replay buffer variant that also stores message-related state

### Evaluation and analysis layer

- `eval.py`
  - greedy evaluation for both baseline and comm checkpoints
  - auto-detects model type from checkpoint metadata

- `checkpoint_sweep.py`
  - evaluates periodic checkpoints
  - selects the best checkpoint per run using greedy `easy_open` performance

- `eval_all_seeds.py`
  - bulk evaluation helper across seeds and maps

- `plot_training.py`
  - single-condition training curve figure

- `plot_comm_training.py`
  - combined training curves for no-comm, comm-4, and comm-16

- `analyze_messages.py`
  - interpretability analysis for saved message logs
  - reports entropy, per-agent usage, temporal usage, and only runs capture/escape analysis when enough failure examples exist

## Folder Structure

Key files and folders:

- `specs.md`
  - the original project target and experiment spec

- `PROGRESS.md`
  - the detailed running research log and current findings

- `envs/`
  - environment wrappers and map definitions

- `models/`
  - baseline and communication networks

- `training/`
  - replay buffers and training helpers

- `results/`
  - checkpoints, eval summaries, plots, analysis outputs, and GIFs

- `scripts/inspect_pursuit.py`
  - early environment-inspection/debug script

## Best Current Findings

The strongest current findings are:

- all three conditions solve `easy_open`
- communication improves mean transfer performance on unseen maps
- `Comm-16` is the strongest communication variant overall
- no-communication runs are more seed-sensitive
- message channels are fully used, but the learned protocol is not symbolically interpretable at the single-symbol level

Important nuance:

- communication is **not** a clean win on every single run or map
- the safest claim is about better average transfer behavior and more stable convergence, not universal dominance

## How Model Comparison Works

Training saves checkpoints every `500` episodes.

Instead of blindly using the final checkpoint, this project uses a **best-checkpoint selection rule**:

1. greedily evaluate each saved checkpoint on `easy_open`
2. choose the checkpoint with the highest capture rate
3. break ties using lower average steps
4. use that selected checkpoint for transfer-map evaluation

This matters because some runs partially degrade late in training, and final-checkpoint-only comparisons can be misleading.

## Message Interpretability

The message-analysis pipeline asks:

- do agents use the whole vocabulary?
- do different agents specialize into different symbol patterns?
- do symbols shift across early, middle, and late parts of an episode?
- do certain symbols correlate with success or failure?

Current answer:

- both communication channels use nearly the full available entropy
- agents do not show strong role differentiation
- symbol usage does not shift much across episode phases
- on the easy training map, success/failure comparison is usually not meaningful because the agents almost never fail

Interpretation:

**the messages appear functional but distributed, not cleanly symbolic**

## Rollout Notes

The rollout GIFs are mainly for presentation and qualitative sanity checks.

One pattern you may notice on obstacle maps is what looks like agents "waiting" near a corridor or bottleneck. That is not automatically a bug. In successful rollouts, the team often holds blocking positions to limit the prey's escape routes and only advances once the prey commits to a lane.

The safest interpretation is:

**the agents sometimes use containment behavior rather than pure chase behavior**

## Quick Start

Use the repo virtual environment:

```bash
source ../.venv/bin/activate
cd marl-comms-research
pip install -r requirements.txt
```

Train the no-communication baseline:

```bash
python train_baseline.py \
  --map easy_open \
  --episodes 5000 \
  --seed 0 \
  --n-catch 1 \
  --distance-reward-scale 0.1 \
  --eps-decay-steps 500000 \
  --warmup-steps 5000 \
  --results-dir results/baseline_v3
```

Train the communication models:

```bash
python train_comm.py \
  --vocab-size 4 \
  --map easy_open \
  --episodes 5000 \
  --seed 0 \
  --distance-reward-scale 0.1 \
  --eps-decay-steps 500000 \
  --warmup-steps 5000 \
  --results-dir results/comm4_v2
```

```bash
python train_comm.py \
  --vocab-size 16 \
  --map easy_open \
  --episodes 5000 \
  --seed 0 \
  --distance-reward-scale 0.1 \
  --eps-decay-steps 500000 \
  --warmup-steps 5000 \
  --results-dir results/comm16_v2
```

Run greedy evaluation:

```bash
python eval.py \
  --checkpoint results/comm16_v2/checkpoints/comm16_final.pt \
  --episodes 200 \
  --seed 99 \
  --output results/comm16_v2/eval_easy_open.json
```

Run the checkpoint sweep:

```bash
python checkpoint_sweep.py --results-dir results --episodes 200 --seed 99
```

Run message analysis:

```bash
python analyze_messages.py \
  --log results/comm16_v2/comm16_msg_log_seed0.json \
  --vocab-size 16 \
  --out-dir results/msg_analysis_16
```

## Recommended Reading Order

If you're new to the project, the easiest path is:

1. read `specs.md` for the original goal
2. read `PROGRESS.md` for the detailed experiment story
3. look through `results/` for the concrete outputs
4. inspect `train_baseline.py`, `train_comm.py`, and `eval.py` for the core pipeline

## Most Useful Artifacts

If you are preparing slides or explaining the project quickly, the most useful outputs are:

- `results/comm_training_curves.png`
- `results/sweep_summary.md`
- `results/all_seeds_raw.json`
- `results/msg_analysis/message_analysis.md`
- `results/msg_analysis_16/message_analysis.md`
- rollout GIFs in `results/`

For the full detailed experiment narrative, use `PROGRESS.md`.
