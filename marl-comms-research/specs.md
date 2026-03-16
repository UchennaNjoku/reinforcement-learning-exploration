# MARL Comms Research Spec

## Project Status

This file freezes the initial problem definition and execution scope for the first phase of the project.

Freeze date: 2026-03-14

## Research Question

In partially observable multi-agent pursuit, does a tiny learned discrete communication channel improve coordination over no communication, and do the learned messages become structured enough to analyze?

## Core Claim

The first paper-scale claim is:

`2-bit discrete communication improves coordination over no communication in pursuit, and approaches the performance of a higher-bandwidth discrete channel.`

## Environment

- Base environment: `pettingzoo.sisl.pursuit_v4`
- API target: parallel environment API
- Initial pursuers: `3`
- Initial evaders/prey: `1`
- Observation setting: local partial observation window
- Target local view: `7 x 7`
- Initial training curriculum: start with `1` easy fixed map
- Expansion target after first win: `3-5` fixed maps

## Agent Setup

- Parameter sharing across pursuers
- Decentralized execution
- Shared replay buffer
- Shared Q-network with agent identity input
- No communication baseline first

## Communication Design

- Primary communication approach: joint-action discrete messaging
- Joint action definition: `(move, message)`
- Base move action count: `5`
- Main message vocabulary: `4` symbols (`2-bit`)
- Higher-bandwidth comparison vocabulary: `16` symbols (`4-bit`)
- Message timing: each agent receives teammate messages from timestep `t-1`

## Experimental Conditions

The first locked comparison conditions are:

1. No communication
2. `4`-symbol discrete communication
3. `16`-symbol discrete communication

## Metrics

- Capture rate
- Average steps to capture
- Escape rate
- Collision rate

## Evaluation Protocol

- Minimum seeds per condition: `3`
- First target: prove the effect on one easy map before expanding
- Evaluation output format: JSON metrics per run
- Artifact expectations for each major checkpoint:
  - saved checkpoint
  - training plot
  - rollout GIF or equivalent visualization

## Scope Cuts

The following are explicitly postponed for now:

- continuous communication
- Gumbel-Softmax communication learning
- centralized critic beyond what is strictly needed
- attention-based communication fusion
- large map suites before a one-map result
- broad ablations before the main comparison works

## First Two Implementation Tasks

1. Stand up `pursuit_v4`
2. Inspect and document raw observations and actions

## Success Criteria For This Freeze

Checkpoint 1 is complete when:

- the environment runs locally
- a reproducible inspection script exists
- raw observation and action structure are documented
- this spec remains stable while we build the first baseline

