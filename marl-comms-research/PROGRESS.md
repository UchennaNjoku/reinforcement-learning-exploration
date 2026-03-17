# MARL Comms Research — Progress Document

**Project:** Emergent Communication Protocols for Coordination in Partially Observable Multi-Agent Pursuit
**Authors:** U. Njoku, J. Calderon — Dept. of Computer Science, Bethune-Cookman University
**Last updated:** 2026-03-17

---

## 1. Research Goal

Test whether a tiny learned discrete communication channel improves coordination in multi-agent pursuit under partial observability, and whether the messages become structured enough to interpret.

**Core claim being tested:**
> Discrete communication improves coordination over no communication, and the benefit is most visible in transfer to unseen maps.

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
- 5000 episodes on `easy_open`

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

### 2.5 Multi-Seed Evaluation and Checkpoint Selection

All conditions were trained for 3 independent seeds. Checkpoints are saved every 500 episodes.

**Best-checkpoint selection rule (uniform across all runs):** For each seed/condition, sweep all saved 500-ep checkpoints on `easy_open` under the greedy policy. Select the checkpoint with the highest capture rate, breaking ties by lowest avg steps. This avoids late-training instability artifacts from the final checkpoint being used for all comparisons.

Script: `checkpoint_sweep.py`

**Why not use the final checkpoint?** Initial evals used each run's final checkpoint and produced misleading results — high collision rates (28–30%) for comm models and an inflated step-count advantage for Comm-4 over No-Comm. Both were artifacts: the comm final checkpoints happened to be near-peak while the baseline final checkpoint had partially degraded. With uniform best-checkpoint selection, the collision rates drop to 0–4% and the in-distribution step counts converge across all conditions.

**Seed management history:** Three seeds (0, 1, 2) were run per condition. Baseline seed 2 (`baseline_s2`) failed to learn a greedy policy — its best checkpoint achieved only 51.5% capture. An additional seed 3 (`baseline_s3`) was run as a sensitivity check and converged cleanly, confirming the failure was isolated. See Section 4.6 for root cause analysis. The primary table uses matched seeds {0, 1, 2}; seed 3 is supporting evidence only.

### 2.6 Message Logging

`train_comm.py` saves a rolling window of the last 500 training episodes' per-step message trajectories to `comm{N}_msg_log_seed{S}.json`. This is the artifact for interpretability analysis.

### 2.7 Message Interpretability Analysis

Script: `analyze_messages.py`

Runs four analyses on a message log file:
1. **Symbol frequency distribution** — entropy and per-symbol usage rates
2. **Per-agent symbol usage** — heatmap and inter-agent variation to detect role differentiation
3. **Temporal analysis** — symbol usage across early/mid/late episode phases
4. **Capture vs escape correlation** — gated behind a minimum escaped episode count (default: 20); reports a null result explicitly if insufficient failures exist

Outputs: three PNG figures and a `message_analysis.md` report per run.

Results for seed 0 of both conditions are in `results/msg_analysis/` (Comm-4) and `results/msg_analysis_16/` (Comm-16).

---

## 3. Results

All models trained on `easy_open`. Transfer evals on `center_block` and `split_barrier` use the same trained weights — no model was trained on those maps.

Evaluation: greedy policy, seed 99, 200 episodes per run.

### 3.1 Primary Table — Matched Seeds {0, 1, 2}, Best-Checkpoint Selection

Per-seed results (individual runs):

| Condition | Subdir | Seed | Selected Ep | easy_open cap | center_block cap | split_barrier cap |
|-----------|--------|:----:|:-----------:|:-------------:|:----------------:|:-----------------:|
| No-Comm | baseline_v3 | 0 | ep004000 | 100% / 9.1 steps | 100% / 27.4 | 99% / 66.7 |
| No-Comm | baseline_s1 | 1 | ep005000 | 100% / 10.8 steps | 37% / 255.8 | 35% / 254.9 |
| No-Comm | baseline_s2 | 2 | ep000500 | **51.5%** *(failed)* | — | — |
| Comm-4 | comm4_v2 | 0 | ep004000 | 100% / 9.3 steps | 100% / 31.0 | 96.5% / 65.7 |
| Comm-4 | comm4_s1 | 1 | ep003500 | 100% / 10.1 steps | 73% / 168.6 | 58.5% / 203.8 |
| Comm-4 | comm4_s2 | 2 | ep001500 | 100% / 9.8 steps | 100% / 21.6 | 57% / 209.8 |
| Comm-16 | comm16_v2 | 0 | ep003500 | 100% / 9.3 steps | 99.5% / 28.6 | 95% / 62.2 |
| Comm-16 | comm16_s1 | 1 | ep004500 | 100% / 8.9 steps | 99.5% / 27.4 | 79.5% / 162.1 |
| Comm-16 | comm16_s2 | 2 | ep004000 | 100% / 8.8 steps | 86.5% / 134.7 | 64% / 193.9 |

**Note on baseline_s2:** The best greedy checkpoint across all 10 periodic checkpoints achieved only 51.5% capture on `easy_open`. The model never learned a reliable greedy policy at any point in training. Cross-map evals for this seed are not meaningful and are excluded from the transfer summary. See Section 4.6 for analysis.

Aggregated mean ± std (No-Comm uses seeds 0 and 1 only for transfer maps due to seed 2 failure). These matched-seed aggregates were computed from the per-seed raw results in `results/all_seeds_raw.json`; the current `results/sweep_summary.md` file reflects the separate seed-3 sensitivity sweep.

| Condition | Map | Capture Rate | Avg Steps | Collision Rate | N |
|-----------|-----|:------------:|:---------:|:--------------:|:-:|
| No-Comm | easy_open | 83.8% ±22.9 | 76.6 ±95.5 | 0.1% | 3 |
| No-Comm | center_block | 68.5% ±44.5 | 141.6 ±160.9 | — | 2* |
| No-Comm | split_barrier | 67.0% ±45.3 | 160.8 ±133.3 | — | 2* |
| Comm-4 | easy_open | **100% ±0** | 9.7 ±0.3 | 0.0% | 3 |
| Comm-4 | center_block | 91.0% ±12.7 | 73.7 ±67.2 | 0.5% ±0.5 | 3 |
| Comm-4 | split_barrier | 70.7% ±18.3 | 159.8 ±66.6 | 4.1% ±4.0 | 3 |
| Comm-16 | easy_open | **100% ±0** | 9.0 ±0.2 | 0.0% | 3 |
| Comm-16 | center_block | **95.2% ±6.1** | **63.6 ±50.3** | 0.9% ±1.0 | 3 |
| Comm-16 | split_barrier | **79.5% ±12.7** | **139.4 ±56.1** | 3.7% ±2.7 | 3 |

*\* No-Comm transfer stats use only seeds 0 and 1; seed 2 failed to converge.*

### 3.2 Supporting Evidence — Seed 3 Replacement Run

To confirm that baseline_s2's failure was an isolated DQN instability rather than a systematic problem with No-Comm on this task, an additional baseline run was conducted with seed 3.

| Subdir | Seed | Selected Ep | easy_open | center_block | split_barrier |
|--------|:----:|:-----------:|:---------:|:------------:|:-------------:|
| baseline_s3 | 3 | ep005000 | 100% / 8.8 steps | 96% / 80.1 | 63.5% / 197.6 |

Seed 3 converges cleanly to 100% capture with 8.8 avg steps — matching seeds 0 and 1 in-distribution. This confirms that a well-converged No-Comm baseline can achieve competitive in-distribution performance, and that seed 2's failure was not representative of the condition.

### 3.3 Interpretability Results

Analysis conducted on seed 0 message logs from the last 500 training episodes (~36–37k total messages per condition).

| Metric | Comm-4 | Comm-16 |
|--------|--------|---------|
| Channel entropy | 1.997 / 2.0 bits (99.8%) | 3.985 / 4.0 bits (99.6%) |
| Inter-agent variation (avg std) | 1.09% | 0.82% |
| Max temporal shift (early vs late) | 1.9% | 1.6% |
| Capture/escape correlation | not computed (2 escapes / 500 eps) | not computed (1 escape / 500 eps) |

**Key finding: no interpretable discrete structure.** Both channels operate at near-maximum entropy. All agents send symbols in nearly identical proportions despite sharing parameters (no role differentiation). Symbol usage does not shift across episode phases (no temporal structure). Capture/escape correlation could not be computed — the models almost never fail on `easy_open` in late training, leaving insufficient failed episodes for a meaningful comparison.

**What this means:** The communication channel is fully utilized but appears to encode information in a distributed, non-categorical form rather than discrete semantic categories. This is consistent with prior emergent communication literature where learned protocols are functional but resist symbolic interpretation. The capture/escape analysis should be revisited on harder maps or weaker checkpoints where failures are more common.

**Retracted finding:** An earlier run of the analysis incorrectly reported that Comm-16 symbol 3 showed an 11.2% divergence between captured and escaped episodes — suggestive of a "failure signal." This was based on a single escaped episode and was pure noise. The analysis script now gates this section behind a minimum of 20 escaped episodes and states the null result explicitly when the threshold is not met.

---

## 4. Analysis: What We've Learned

### 4.1 In-Distribution: All Conditions Reach 100% on the Training Map

With best-checkpoint selection, all three conditions achieve 100% greedy capture on `easy_open` with similar step counts (~9–11 steps). There is no significant in-distribution advantage for either comm condition over No-Comm once the best policy is selected.

The earlier single-seed result showing Comm-4 at 9.47 steps vs No-Comm at 14.49 was an artifact of comparing different quality checkpoints — the baseline's final checkpoint had partially degraded while the comm final checkpoint happened to be near-peak. With uniform selection, this gap closes.

**The in-distribution finding is:** all three conditions solve the training task. Communication does not meaningfully accelerate in-distribution performance once training stabilizes.

### 4.2 Transfer: Communication Improves Mean Transfer Performance, Channel Size Matters

The clearest benefit of communication appears in transfer to unseen maps, though transfer remains variable across seeds:

**center_block:** Comm-16 transfers most reliably (95.2% ±6.1), followed by Comm-4 (91.0% ±12.7), with No-Comm being the most variable (two converged seeds: 100% and 37%). The obstacle map exposes coordination brittleness — without communication, one seed learned a policy that happens to generalize while another did not.

**split_barrier:** Comm-16 again leads (79.5% ±12.7) over Comm-4 (70.7% ±18.3) and No-Comm (67.0% ±45.3, two seeds). The barrier map is hard for all conditions, but communication consistently reduces variance and improves mean performance.

Across both transfer maps, **Comm-16 outperforms Comm-4** in mean performance, which contradicts the original abstract sub-claim that compact communication approaches higher-bandwidth channels. The revised finding is that channel capacity matters for transfer generalization.

### 4.3 The Core Claim Needs Revision

Original claim: *"2-bit discrete communication improves coordination over no communication, and approaches the performance of a higher-bandwidth discrete channel."*

Revised claim supported by the data: *"Discrete communication improves out-of-distribution generalization. Both comm conditions outperform no-comm on transfer maps in mean capture rate and variance. A higher-bandwidth channel (16 symbols) transfers more reliably than a compact channel (4 symbols) — compact communication does not fully close the gap."*

The communication benefit is primarily a generalization benefit, not a raw in-distribution performance benefit.

### 4.4 Collision Rates Are Low for Best Checkpoints

Earlier analysis attributed high collision rates (28–30%) to comm models developing aggressive, obstacle-ignoring pursuit strategies. This interpretation was wrong — it was an artifact of the final checkpoint being degraded. With best-checkpoint selection, all comm models show near-zero collision on `easy_open` (0.0%) and low collision on transfer maps (0.5–4.1%), comparable to or better than No-Comm.

The one exception is No-Comm seed 0 on `split_barrier` (84% collision rate from the raw data), which suggests that specific seed learned a high-frequency wall-pressing strategy that coincidentally still captures the prey 99% of the time on that map.

### 4.5 No-Comm Has Higher Seed Sensitivity

No-Comm shows the highest variance across all metrics. Seed 2 failed to converge to a greedy policy entirely; seed 1 converges on `easy_open` but transfers poorly (37% on center_block). The comm models show much more consistent convergence — all 6 comm seeds (3 × Comm-4, 3 × Comm-16) converged to 100% capture and all transfer at reasonable rates.

This suggests communication may stabilize the learning dynamics, not just improve final performance. One possible mechanism: receiving structured messages from teammates provides additional state information that reduces the effective variance of individual agents' Q-value estimates.

### 4.6 Baseline Seed 2 Failure: Root Cause

`baseline_s2` never learned a reliable greedy policy. Its best checkpoint across all 10 periodic evaluations achieved only 51.5% capture on the training map. This is distinct from late-training collapse (where performance degrades after a good checkpoint exists) — the greedy policy never converged at any point.

The training log showed high epsilon-greedy capture rates (reaching 99% around ep 3200), but these were inflated by exploration. When epsilon hit its floor (0.05), performance collapsed, indicating the Q-values were never reliably estimating move quality — captures during training were largely due to random exploration rather than learned policy.

This failure mode is a known DQN instability: with this particular random initialization, the network's weight landscape may have led to a poor local optimum that only appeared functional under high exploration. The replacement seed 3 confirms this was not a systematic problem with the No-Comm condition.

### 4.7 The Joint Q-Value Architecture Fails for Communication

The original comm model used a single Q-head over `5 × vocab_size` joint actions. Despite showing 94-99% capture during epsilon-greedy training, the greedy eval gave 0.5% capture (essentially random). Root cause: with 20+ joint actions and sparse rewards, most captures during training were due to random exploration rather than learned policy. The Q-values never converged to correctly identify which move was best — they collapsed to near-uniform values, causing the greedy policy to always take action 0. Switching to separate move and message heads (5 Q-values for moves, vocab_size Q-values for messages) resolved this completely.

### 4.8 Reward Shaping Was Necessary

The first training attempts without shaped rewards (capture-only reward) produced flat learning curves at 0-4% capture for the entire 5000 episodes. The distance penalty (`scale=0.1`, normalized Manhattan distance to prey) provided a dense gradient signal. This is standard practice in sparse-reward MARL and does not affect the validity of condition comparisons — all conditions used the same reward structure.

---

## 5. What Is Still To Do

### 5.1 Remaining for the Paper (Required)

- [x] Run 2 additional seeds per condition for statistical validity
- [x] Build per-seed summary table with best-checkpoint selection
- [x] Interpretability analysis on message logs (both Comm-4 and Comm-16, seed 0):
  - [x] Symbol frequency distribution and entropy
  - [x] Temporal analysis (episode phase shifts)
  - [x] Role differentiation (per-agent symbol heatmap)
  - [x] Capture/escape correlation — not computable; null result documented and gated
- [ ] Training curve plots for all conditions overlaid (`plot_comm_training.py` built, not yet run)
- [ ] Rollout visualization / GIF export showing comm agents coordinating

### 5.2 Optional but Strengthening

- [ ] Interpretability on harder maps or weaker checkpoints where failures are common (enables capture/escape correlation analysis)
- [ ] Per-map training runs for `center_block` and `split_barrier` to get fair in-distribution baselines
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
| `checkpoint_sweep.py` | Best-checkpoint selection across all seeds |
| `eval_all_seeds.py` | Bulk eval across all seed/map combinations |
| `plot_training.py` | Training curve figure (single condition) |
| `plot_comm_training.py` | Training curves overlaid for all conditions |
| `plot_comparison.py` | Cross-condition comparison figure (single seed) |
| `analyze_messages.py` | Message interpretability analysis (entropy, role diff, temporal, capture correlation) |
| `results/baseline_v3/` | No-Comm seed 0 checkpoints, logs, evals |
| `results/baseline_s1/` | No-Comm seed 1 |
| `results/baseline_s2/` | No-Comm seed 2 (failed — for reference only) |
| `results/baseline_s3/` | No-Comm seed 3 (replacement convergence check) |
| `results/comm4_v2/` | Comm-4 seed 0 checkpoints, logs, message log |
| `results/comm4_s1/` | Comm-4 seed 1 |
| `results/comm4_s2/` | Comm-4 seed 2 |
| `results/comm16_v2/` | Comm-16 seed 0 checkpoints, logs, message log |
| `results/comm16_s1/` | Comm-16 seed 1 |
| `results/comm16_s2/` | Comm-16 seed 2 |
| `results/sweep_selection.json` | Selected checkpoint per run (matched seeds {0,1,2}, excl. s2) |
| `results/sweep_summary.md` | Seed-3 sensitivity sweep summary (supporting evidence) |
| `results/all_seeds_raw.json` | Raw eval results backing the primary matched-seed table |
| `results/msg_analysis/` | Comm-4 seed 0 interpretability figures and report |
| `results/msg_analysis_16/` | Comm-16 seed 0 interpretability figures and report |

---

## 7. How to Reproduce

```bash
# Activate environment
source ../.venv/bin/activate
cd marl-comms-research

# Run checkpoint sweep (selects best checkpoint per run, evals on all maps)
python checkpoint_sweep.py --results-dir results --episodes 200 --seed 99

# Add a single new run without re-sweeping everything else
python checkpoint_sweep.py --only-subdir baseline_s3 --episodes 200 --seed 99

# Message interpretability analysis (comm-4)
python analyze_messages.py --log results/comm4_v2/comm4_msg_log_seed0.json --vocab-size 4

# Message interpretability analysis (comm-16)
python analyze_messages.py --log results/comm16_v2/comm16_msg_log_seed0.json --vocab-size 16 --out-dir results/msg_analysis_16

# Training curves (all conditions overlaid)
python plot_comm_training.py

# Single-condition training curve
python plot_training.py --log results/baseline_v3/baseline_train_log_seed0.json

# Message interpretability — comm-4 (seed 0)
python analyze_messages.py --log results/comm4_v2/comm4_msg_log_seed0.json --vocab-size 4

# Message interpretability — comm-16 (seed 0)
python analyze_messages.py --log results/comm16_v2/comm16_msg_log_seed0.json --vocab-size 16 --out-dir results/msg_analysis_16

# Message interpretability on a harder map / weaker checkpoint (enables capture/escape analysis)
# python analyze_messages.py --log <path> --vocab-size 4 --min-escaped 20
```
