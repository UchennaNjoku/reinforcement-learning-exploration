"""
Surround Pursuit — IPPO Training (EPyMARL)
==========================================
Run this after cloning your repo on Kaggle:

    !python /kaggle/working/reinforcement-learning-exploration/possible_extensions/kaggle/run_training.py

Optional args (set via env vars before running):
    SEED=0          training seed (default 0)
    T_MAX=5000000   total env steps (default 5M, ~4-5h on T4)
    SANITY=1        set to 1 to do a 1000-step sanity check only
"""

import os
import sys
import subprocess
import textwrap
from pathlib import Path

# ── Config (override via env vars) ─────────────────────────────────────────
SEED    = int(os.environ.get("SEED",    0))
T_MAX   = int(os.environ.get("T_MAX",  5_000_000))
SANITY  = os.environ.get("SANITY", "0") == "1"

WORK    = Path("/kaggle/working")
EPYMARL = WORK / "epymarl"

def run(cmd, **kwargs):
    print(f"\n>>> {cmd}\n")
    result = subprocess.run(cmd, shell=True, **kwargs)
    if result.returncode != 0:
        print(f"ERROR: command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

# ── Step 1: Install dependencies ────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Installing dependencies")
print("=" * 60)
run("pip install -q pettingzoo[sisl]==1.24.3 sacred pymongo==3.12.3 pygame")

# ── Step 2: Clone EPyMARL ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Cloning EPyMARL")
print("=" * 60)
if EPYMARL.exists():
    print("EPyMARL already cloned, skipping.")
else:
    run(f"git clone https://github.com/uoe-agents/epymarl.git {EPYMARL}")

run(f"pip install -q -r {EPYMARL}/requirements.txt")

# ── Step 3: Verify pursuit_v4 loads ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Verifying pursuit_v4")
print("=" * 60)
from pettingzoo.sisl import pursuit_v4
env = pursuit_v4.parallel_env(
    x_size=16, y_size=16, n_pursuers=8, n_evaders=30,
    obs_range=7, n_catch=2, surround=True, shared_reward=True,
    freeze_evaders=False, tag_reward=0.01, catch_reward=5.0,
    urgency_reward=-0.1, max_cycles=500, render_mode=None
)
obs, _ = env.reset()
print(f"Agents: {len(env.agents)}, obs shape: {list(obs.values())[0].shape}")
env.close()
print("pursuit_v4 OK.")

# ── Step 4: Write environment config ────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Writing configs")
print("=" * 60)

env_config = textwrap.dedent("""
    env: "gymma"
    env_args:
      key: "pz-pursuit-v4"
      time_limit: 500
      pretrained_wrapper: null
""").strip()

(EPYMARL / "src/config/envs/pursuit_surround.yaml").write_text(env_config)
print("Env config written.")

# ── Step 5: Write IPPO config ────────────────────────────────────────────────
ippo_config = textwrap.dedent(f"""
    action_selector: "softmax"
    softmax_temp: 1.0
    use_rnn: True
    rnn_hidden_dim: 64
    epochs: 4
    eps_clip: 0.1
    standardise_rewards: True
    standardise_returns: False
    use_gae: True
    gae_lambda: 0.95
    use_value_norm: True
    lr: 0.0005
    optim_alpha: 0.99
    optim_eps: 0.00001
    grad_norm_clip: 10
    batch_size_run: 8
    batch_size: 32
    buffer_size: 32
    entropy_coef: 0.01
    use_huber_loss: True
    huber_delta: 10.0
    agent: "rnn"
    mac: "ppo_mac"
    learner: "ppo_learner"
    name: "ippo"
    gamma: 0.99
    log_interval: 10000
    runner_log_interval: 10000
    learner_log_interval: 10000
    t_max: {T_MAX}
    save_model: True
    save_model_interval: 500000
    checkpoint_path: ""
    evaluate: False
    test_nepisode: 20
    test_interval: 50000
    test_greedy: True
    runner: "parallel"
    env_runner_split: null
""").strip()

(EPYMARL / "src/config/algs/ippo_pursuit.yaml").write_text(ippo_config)
print("IPPO config written.")

# ── Step 6: Write the pursuit gym wrapper ────────────────────────────────────
wrapper_code = textwrap.dedent("""
    from gymnasium.envs.registration import register
    from pettingzoo.sisl import pursuit_v4
    import gymnasium as gym
    import numpy as np

    class PursuitGymWrapper(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            self.env = pursuit_v4.parallel_env(
                x_size=16, y_size=16, n_pursuers=8, n_evaders=30,
                obs_range=7, n_catch=2, surround=True, shared_reward=True,
                freeze_evaders=False, tag_reward=0.01, catch_reward=5.0,
                urgency_reward=-0.1, max_cycles=500, render_mode=None
            )
            self.n_agents = 8
            sample = self.env.possible_agents[0]
            self.observation_space = self.env.observation_space(sample)
            self.action_space      = self.env.action_space(sample)

        def reset(self, seed=None, options=None):
            obs, info = self.env.reset(seed=seed)
            return np.stack([obs[a] for a in self.env.possible_agents]), info

        def step(self, actions):
            action_dict = {a: int(actions[i]) for i, a in enumerate(self.env.agents)}
            obs, rewards, terms, truncs, infos = self.env.step(action_dict)
            obs_list  = [obs.get(a, np.zeros(self.observation_space.shape))
                         for a in self.env.possible_agents]
            rew_list  = [rewards.get(a, 0.0) for a in self.env.possible_agents]
            term_list = [terms.get(a, False)  for a in self.env.possible_agents]
            trunc_list= [truncs.get(a, False) for a in self.env.possible_agents]
            done = all(term_list) or all(trunc_list) or not self.env.agents
            return np.stack(obs_list), np.array(rew_list), done, False, infos

        def render(self): pass
        def close(self): self.env.close()

    register(id="pz-pursuit-v4", entry_point="pursuit_gym_wrapper:PursuitGymWrapper")
""").strip()

(EPYMARL / "src/pursuit_gym_wrapper.py").write_text(wrapper_code)
print("Gym wrapper written.")

# ── Step 7: Patch main.py to import wrapper ──────────────────────────────────
main_path = EPYMARL / "src/main.py"
main_content = main_path.read_text()
import_line = 'import sys; sys.path.insert(0, "/kaggle/working/epymarl/src"); import pursuit_gym_wrapper\n'
if "pursuit_gym_wrapper" not in main_content:
    main_path.write_text(import_line + main_content)
    print("main.py patched.")
else:
    print("main.py already patched.")

# ── Step 8: Run training ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
if SANITY:
    print("STEP 5 — Sanity check (1000 steps)")
    print("=" * 60)
    run(
        f"cd {EPYMARL}/src && python main.py "
        f"--config=ippo_pursuit --env-config=pursuit_surround "
        f"with t_max=1000 save_model=False test_interval=1000 log_interval=500"
    )
    print("\nSanity check passed. Re-run without SANITY=1 for full training.")
else:
    print(f"STEP 5 — Full training (seed={SEED}, t_max={T_MAX:,})")
    print("=" * 60)
    print("Target reward: ~633 (FLAIRS 2024 baseline)")
    print("Estimated time: 4-5h on T4 GPU for 5M steps\n")
    run(
        f"cd {EPYMARL}/src && python main.py "
        f"--config=ippo_pursuit --env-config=pursuit_surround "
        f"with seed={SEED} t_max={T_MAX}"
    )

# ── Step 9: Save results ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — Saving results")
print("=" * 60)
import shutil
results_dir = EPYMARL / "results"
if results_dir.exists():
    out = WORK / f"results_seed{SEED}.zip"
    shutil.make_archive(str(out).replace(".zip", ""), "zip", str(results_dir))
    size_mb = out.stat().st_size / 1e6
    print(f"Saved {out.name} ({size_mb:.1f} MB)")
    print("Download from the Kaggle output panel on the right.")
else:
    print("No results to save.")
