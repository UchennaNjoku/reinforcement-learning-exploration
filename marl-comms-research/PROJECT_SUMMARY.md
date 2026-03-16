# MARL Comms Research Project Summary

## What This Project Is

This project is a focused multi-agent reinforcement learning research build around one central question:

**In a partially observable pursuit task, does a very small learned communication channel help agents coordinate better than no communication, and do those learned messages become meaningful enough to analyze?**

This is not a general MARL playground. It is a paper-oriented research implementation meant to support the abstract-level claim that compact communication can improve coordination in multi-agent pursuit under partial observability.

## What We Are Working Toward

The current target claim is:

`2-bit discrete communication improves coordination over no communication, and can approach the performance of a higher-bandwidth discrete communication channel.`

That means the project eventually needs to deliver four things:

1. A stable partially observable pursuit environment.
2. A no-communication baseline that trains and evaluates cleanly.
3. A communicating model that improves on the baseline.
4. Analysis showing the messages have some structure or interpretability.

## Important Abstract Alignment Note

The original abstract direction described a CTDE-style setup with continuous communication and discrete communication using Gumbel-Softmax. The implementation path we have frozen for now is simpler and much more buildable:

- shared-parameter DQN-style training
- decentralized execution
- discrete communication
- joint action selection over `(move, message)`

So the practical implementation path currently leads toward:

- no communication baseline
- `4`-symbol (`2-bit`) discrete communication
- `16`-symbol (`4-bit`) discrete communication

If the paper abstract later needs to match the code more closely, this is the point to revise toward.

## What Has Been Frozen So Far

The initial project freeze lives in [specs.md](/Users/uchen/Code/reinforcement-learning/reinforcement-learning-exploration/marl-comms-research/specs.md).

The key frozen choices are:

- environment base: `pettingzoo.sisl.pursuit_v4`
- API style: parallel environment API
- pursuers: `3`
- prey/evaders: `1`
- observation window: `7 x 7` local view
- move action count: `5`
- main communication vocabulary: `4` symbols
- higher-bandwidth comparison vocabulary: `16` symbols
- evaluation metrics:
  - capture rate
  - average steps to capture
  - escape rate
  - collision rate
- seed count target: minimum `3`
- first experimental scope: prove the effect on one easy map before expanding

## What Is Already Built

### 1. Environment inspection

We successfully stood up `pursuit_v4` and inspected the raw action and observation structure.

Inspection files:

- [inspect_pursuit.py](/Users/uchen/Code/reinforcement-learning/reinforcement-learning-exploration/marl-comms-research/scripts/inspect_pursuit.py)
- [pursuit_v4_inspection.json](/Users/uchen/Code/reinforcement-learning/reinforcement-learning-exploration/marl-comms-research/artifacts/pursuit_v4_inspection.json)

What we learned:

- active agents are `pursuer_0`, `pursuer_1`, `pursuer_2`
- each pursuer action space is `Discrete(5)`
- each pursuer observation space is `Box(0.0, 3.0, (7, 7, 3), float32)`
- reset observations are raw arrays of shape `(7, 7, 3)`

### 2. Basic demo scripts

There are now simple scripts that make the environment easy to run and inspect:

- [run_pursuit_demo.py](/Users/uchen/Code/reinforcement-learning/reinforcement-learning-exploration/marl-comms-research/scripts/run_pursuit_demo.py)
- [main.py](/Users/uchen/Code/reinforcement-learning/reinforcement-learning-exploration/marl-comms-research/main.py)

`main.py` is the best current starting point for the project.

### 3. First wrapper around pursuit_v4

We added a research-facing environment wrapper at [fixed_pursuit.py](/Users/uchen/Code/reinforcement-learning/reinforcement-learning-exploration/marl-comms-research/envs/fixed_pursuit.py).

This wrapper currently gives us:

- fixed `3 pursuers + 1 prey`
- fixed map presets
- fixed start positions
- cleaner reward defaults
- a simpler, more controlled research interface
- extra step info like position, blocked move, and evaders remaining

## Current Environment Setup

The wrapper currently supports these fixed map presets:

- `easy_open`
- `center_block`
- `split_barrier`

The reward defaults are intentionally simpler than raw pursuit defaults:

- `catch_reward = 1.0`
- `tag_reward = 0.0`
- `step_penalty = -0.01`

The point of these reward settings is to make the first baseline easier to reason about and easier to compare later when communication is added.

## What The Current Codebase Looks Like

The main workspace lives in [marl-comms-research](/Users/uchen/Code/reinforcement-learning/reinforcement-learning-exploration/marl-comms-research).

Important files:

- [specs.md](/Users/uchen/Code/reinforcement-learning/reinforcement-learning-exploration/marl-comms-research/specs.md)
  - frozen problem definition and experimental scope
- [main.py](/Users/uchen/Code/reinforcement-learning/reinforcement-learning-exploration/marl-comms-research/main.py)
  - current simple entry point for running the environment
- [envs/fixed_pursuit.py](/Users/uchen/Code/reinforcement-learning/reinforcement-learning-exploration/marl-comms-research/envs/fixed_pursuit.py)
  - wrapper that fixes counts, maps, starts, and reward defaults
- [scripts/inspect_pursuit.py](/Users/uchen/Code/reinforcement-learning/reinforcement-learning-exploration/marl-comms-research/scripts/inspect_pursuit.py)
  - raw observation/action inspection
- [scripts/run_pursuit_demo.py](/Users/uchen/Code/reinforcement-learning/reinforcement-learning-exploration/marl-comms-research/scripts/run_pursuit_demo.py)
  - simple demo runner
- [requirements.txt](/Users/uchen/Code/reinforcement-learning/reinforcement-learning-exploration/marl-comms-research/requirements.txt)
  - pinned local project dependencies

## What Is Not Built Yet

The actual research pipeline has not started yet. In particular, these pieces are still missing:

- no-communication training loop
- replay buffer
- shared Q-network
- agent identity input / embedding
- target network and checkpointing
- evaluation script that writes JSON metrics
- plots for training curves
- rollout GIF export
- communication model
- message logging and interpretability analysis

So the project is currently in the **environment and infrastructure setup phase**, not the training/results phase.

## Recommended Next Step

Yes, the next major step is to begin training and testing, but the correct first training step is very specific:

**Build the no-communication baseline first.**

That means:

1. Create a training script for shared-parameter Q-learning or DQN-style learning over the wrapped pursuit environment.
2. Use local observation only.
3. Train on one easy map first.
4. Build evaluation before adding communication.
5. Save metrics and at least one checkpoint.

The goal is not yet to make communication work. The goal is to get a stable baseline that:

- runs end to end
- beats random behavior
- produces reproducible metrics

Without that, communication will be much harder to debug.

## Suggested Next Implementation Order

The clean order from here is:

1. `eval.py`
2. no-communication training loop
3. checkpoint saving/loading
4. training plots
5. rollout visualization
6. communication-enabled action head
7. message logging
8. multi-condition comparison
9. interpretability/probing analysis

## Planned Research Conditions

The planned comparison remains:

1. No communication
2. `4`-symbol communication
3. `16`-symbol communication

The intended result pattern is:

`2-bit comm ≈ higher-bandwidth comm > no comm`

That is the core paper-facing story the project is aiming to test.

## How To Run The Current Environment

Activate the repo environment:

```bash
source /Users/uchen/Code/reinforcement-learning/reinforcement-learning-exploration/.venv/bin/activate
```

Run the current research entry point:

```bash
python /Users/uchen/Code/reinforcement-learning/reinforcement-learning-exploration/marl-comms-research/main.py
```

Run headless:

```bash
python /Users/uchen/Code/reinforcement-learning/reinforcement-learning-exploration/marl-comms-research/main.py --headless --map center_block --steps 50
```

Inspect the raw PettingZoo environment:

```bash
python /Users/uchen/Code/reinforcement-learning/reinforcement-learning-exploration/marl-comms-research/scripts/inspect_pursuit.py
```

## One-Screen Handoff Summary

If someone new opens this project, the shortest accurate description is:

This is a MARL research build for testing whether tiny discrete communication helps coordination in partially observable pursuit. The environment base is PettingZoo `pursuit_v4`, but we now run it through a wrapper that fixes the task to `3` pursuers, `1` prey, controlled maps, fixed starts, and cleaner rewards. The environment is running and inspected. The next real milestone is to build the no-communication baseline training and evaluation pipeline before adding any communication.

