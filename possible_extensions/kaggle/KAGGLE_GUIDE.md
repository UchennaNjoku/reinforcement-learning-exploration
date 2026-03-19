# Kaggle Training Guide — Surround Pursuit with EPyMARL + IPPO

## Before you start

- Kaggle gives you a **T4 GPU**, ~30GB disk, and **12 hours** max runtime per session
- IPPO on pursuit_v4 with 8 agents × 30 evaders × 500 steps will take roughly **4–8 hours**
  to reach meaningful convergence (2–5M environment steps)
- You do NOT need to upload any files — everything is installed and cloned inside the notebook

---

## Step 1 — Create the Kaggle notebook

1. Go to kaggle.com → **+ New Notebook**
2. In the top-right settings panel:
   - **Accelerator:** GPU T4 x1
   - **Internet:** ON (required to clone EPyMARL)
   - **Persistence:** Files only
3. Change notebook type to **Script** (not notebook) for cleaner output — optional but recommended

---

## Step 2 — Paste the cells from `setup.ipynb`

The notebook is divided into clearly labeled cells. Paste them in order.
Each cell is marked with `# ── CELL N ──` at the top.

See `kaggle/setup.ipynb` for the full cell contents.

---

## Step 3 — Monitor training

EPyMARL logs to console and saves results to:
```
/kaggle/working/epymarl/results/
```

Key metrics to watch in the logs:
- `episode_reward_mean` — should increase over time toward ~633
- `ep_length_mean` — should decrease as agents get faster at surrounding evaders
- `win_rate` or `capture_rate` — fraction of evaders captured per episode

If reward is flat after 500k steps, the run may have landed on a bad seed. Stop and rerun.

---

## Step 4 — Save results before session ends

Kaggle deletes `/kaggle/working/` when the session ends unless you explicitly save outputs.

Add this at the END of your notebook to zip and preserve results:

```python
import shutil, os
shutil.make_archive('/kaggle/working/results_backup', 'zip',
                    '/kaggle/working/epymarl/results')
print("Saved to /kaggle/working/results_backup.zip")
```

Then download `results_backup.zip` from the Kaggle output panel on the right.

---

## Step 5 — Resume training (if 12h runs out)

EPyMARL saves checkpoints. To resume:
1. Download the results folder (see Step 4)
2. Upload it as a Kaggle dataset
3. In a new notebook, mount the dataset and copy checkpoints back:
   ```python
   import shutil
   shutil.copytree('/kaggle/input/your-dataset-name/results',
                   '/kaggle/working/epymarl/results')
   ```
4. Add `--checkpoint_path` to the training command (see EPyMARL docs)

---

## Expected training timeline (T4 GPU estimate)

| Steps | Wall time | Expected reward |
|-------|-----------|----------------|
| 500k  | ~30 min   | 50–200 (early learning) |
| 1M    | ~1 hr     | 200–400 |
| 2M    | ~2 hr     | 400–550 |
| 5M    | ~5 hr     | 550–650 (near target) |
| 10M   | ~10 hr    | 620–680 (target: 633+) |

These are rough estimates. Variance across seeds is high — run at least 2 seeds.

---

## Troubleshooting

**"No module named pettingzoo.sisl"**
→ Run: `pip install pettingzoo[sisl]` and retry

**"AECEnv is not a ParallelEnv"**
→ EPyMARL's wrapper needs the parallel version. Change the env key to use
  `pursuit_v4.parallel_env()` instead of `.env()`. See setup.ipynb Cell 4 for the patch.

**Reward stays at 0 for 1M+ steps**
→ The catch_reward=5.0 only fires on full encirclement. Make sure `tag_reward=0.01`
  is included — this provides a small dense signal that kick-starts learning.
  If still stuck, try `surround=False` first to verify the pipeline works, then switch back.

**Out of memory**
→ Reduce `batch_size_run` in the IPPO config from 32 to 16, or reduce `buffer_size`.

**Session about to hit 12h limit**
→ EPyMARL saves every N steps automatically. The checkpoint will be in
  `results/sacred/1/models/`. Download before session ends.
