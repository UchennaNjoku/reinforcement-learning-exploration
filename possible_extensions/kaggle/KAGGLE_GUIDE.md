# Kaggle Training Guide — Surround Pursuit with EPyMARL + IPPO

## Setup (one time)

On Kaggle, make sure:
- **Accelerator:** GPU T4 x1
- **Internet:** ON

## Running training — 3 commands total

```python
# Cell 1 — clone your repo (skip if already cloned)
!git clone https://github.com/UchennaNjoku/reinforcement-learning-exploration /kaggle/working/repo

# Cell 2 — sanity check first (1000 steps, ~2 min)
!SANITY=1 python /kaggle/working/repo/possible_extensions/kaggle/run_training.py

# Cell 3 — full training run (~4-5h on T4 for 5M steps)
!python /kaggle/working/repo/possible_extensions/kaggle/run_training.py
```

That's it. The script handles everything:
installs dependencies, clones EPyMARL, writes configs, registers the environment, runs training, and saves a zip of results to the output panel.

## Options

Override defaults with environment variables:

```python
# Different seed
!SEED=1 python /kaggle/working/repo/possible_extensions/kaggle/run_training.py

# Longer run (10M steps, ~8-10h — needed to hit 633+ reward target)
!T_MAX=10000000 python /kaggle/working/repo/possible_extensions/kaggle/run_training.py

# Sanity check only (1000 steps, confirms everything works)
!SANITY=1 python /kaggle/working/repo/possible_extensions/kaggle/run_training.py
```

## Expected training timeline (T4 GPU)

| Steps | Wall time | Expected reward |
|-------|-----------|----------------|
| 500k  | ~30 min   | 50–200 |
| 1M    | ~1 hr     | 200–400 |
| 2M    | ~2 hr     | 400–550 |
| 5M    | ~5 hr     | 550–650 |
| 10M   | ~10 hr    | 620–680 (target: 633+) |

## Results

After training finishes, `results_seed0.zip` will appear in the Kaggle output panel.
Download it and put it in `possible_extensions/results/` in your local repo.

## Troubleshooting

**"No module named pettingzoo.sisl"**
→ The pip install step failed. Re-run Cell 2 (sanity check) which will retry.

**Reward stuck at 0 for 1M+ steps**
→ `tag_reward=0.01` is already set to give a dense signal. If still stuck,
  the run hit a bad seed — stop and rerun with `SEED=1`.

**Out of memory**
→ Reduce batch size in the script: change `batch_size_run: 8` to `4`.

**Session about to hit 12h**
→ The script auto-saves a zip on completion. If training is still going,
  manually run: `import shutil; shutil.make_archive('/kaggle/working/partial_results', 'zip', '/kaggle/working/epymarl/results')`
