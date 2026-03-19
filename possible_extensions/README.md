# Possible Extension: Surround-Mode Pursuit with More Agents

## What this is

An extension of the MARL comms research project using a harder pursuit configuration:

| Setting | Current project | This extension |
|---------|----------------|----------------|
| surround | False (overlap capture) | **True** (must encircle evader) |
| n_pursuers | 3 | **8** |
| n_evaders | 1 | **30** |
| Algorithm | DQN (custom) | **IPPO via EPyMARL** |
| Grid | 16×16 | 16×16 |

## Why surround=True is harder

With `surround=False`, any pursuer overlapping the evader's cell counts as a capture.
With `surround=True`, the evader must be completely enclosed on all 4 cardinal sides
(by pursuers or walls) before it is removed. This requires genuine coordinated encirclement —
not just chasing.

## Why IPPO instead of DQN

DQN with 8 agents and 30 evaders over 500 steps produces very sparse rewards per agent.
IPPO (Independent PPO with parameter sharing) handles this better:
- PPO's clipped surrogate objective is more stable under sparse rewards
- Actor-critic provides a dense value baseline even before captures happen
- Parameter sharing across 8 homogeneous pursuers still applies

## Published benchmark target

FLAIRS 2024 (arXiv:2404.05840) tested pursuit_v4 default config with surround=True:
- Baseline best reward: **633.6**
- Attention-augmented method: **673.7**

That is your training target.

## Files in this folder

```
possible_extensions/
├── README.md                       ← this file
├── configs/
│   └── pursuit_surround.yaml       ← EPyMARL environment config
└── kaggle/
    ├── setup.ipynb                 ← Kaggle notebook (paste cells into Kaggle)
    └── KAGGLE_GUIDE.md             ← Step-by-step Kaggle instructions
```

## Framework used

**EPyMARL** — https://github.com/uoe-agents/epymarl
- Most mature MARL framework with PettingZoo support
- 701 GitHub stars, NeurIPS 2021 paper
- IPPO, MAPPO, QMIX, and 10+ other algorithms included
