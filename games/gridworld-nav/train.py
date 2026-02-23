"""DQN Training for GridWorld Navigation (refactored).

This is a *thin* CLI entry point.

All of the "beef" lives in `training/`:
  - replay buffer:     training/replay.py
  - encoder:           training/encoder.py
  - network:           training/network.py
  - distance map:      training/distance.py
  - reward shaping:    training/reward.py
  - epsilon schedule:  training/schedules.py
  - evaluation:        training/eval.py
  - checkpoints:       training/checkpoint.py
  - warmup:            training/warmup.py
  - core loop:         training/core.py

So when you read this file, you should mostly see:
  1) parse args
  2) create models + buffer
  3) call train_single() or train_curriculum()

Author: Chenna
"""

from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np

try:
    import torch
    import torch.optim as optim
except ImportError:
    print("PyTorch is required. Install with: pip install torch")
    raise

from config import ENV_CONFIG, AGENT_CONFIG, TRAIN_CONFIG, PATHS

from training.checkpoint import load_checkpoint, save_checkpoint
from training.core import TrainState, train_phase
from training.encoder import StateEncoder
from training.eval import evaluate
from training.network import QNetwork
from training.replay import ReplayBuffer
from training.warmup import warmup_buffer


# ─────────────────────────────────────────────────────────────────────────────
# Single-phase runner
# ─────────────────────────────────────────────────────────────────────────────

def train_single(args: argparse.Namespace) -> None:
    """Standard single-phase DQN training."""

    # --- Reproducibility seeds ----------------------------------------------
    # These ensure that (as much as possible) your random choices repeat.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Device --------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # --- Encoder + networks --------------------------------------------------
    encoder = StateEncoder(args.size)

    q_net = QNetwork(encoder.feature_dim, 4).to(device)
    target_net = QNetwork(encoder.feature_dim, 4).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)

    # --- Replay buffer -------------------------------------------------------
    replay = ReplayBuffer(args.replay_size)

    # --- Resume (optional) ---------------------------------------------------
    if args.resume and os.path.exists(args.resume):
        _ = load_checkpoint(args.resume, q_net, target_net, optimizer, device)
        print(f"  Resumed from {args.resume}")

    ts = TrainState(
        q_net=q_net,
        target_net=target_net,
        optimizer=optimizer,
        replay=replay,
        encoder=encoder,
        device=device,
    )

    # --- Pretty run header ---------------------------------------------------
    param_count = sum(p.numel() for p in q_net.parameters())
    print("=" * 70)
    print("  DQN TRAINING — GridWorld Navigation (Single Phase)")
    print("=" * 70)
    print(f"  Device:            {device}")
    print(f"  State dim:         {encoder.feature_dim}")
    print(f"  Network params:    {param_count:,}")
    print(f"  Grid:              {args.size}×{args.size} ({args.size-2}×{args.size-2} playable)")
    print(f"  Wall length:       {args.wall_length}")
    print(f"  Episodes:          {args.episodes}")
    print(f"  Reward shaping:    {'ON' if not args.no_reward_shaping else 'OFF'}")
    print(f"  Shaping weight:    {args.shaping_weight}")
    print(f"  Learning rate:     {args.lr}")
    print(f"  Double DQN:        ON")
    print(f"  Epsilon:           {args.eps_start} → {args.eps_end} over {args.eps_decay_episodes} eps")
    print("=" * 70 + "\n")

    # --- Warmup --------------------------------------------------------------
    print(f"  Warming up ({args.warmup_steps} random steps)...", end=" ", flush=True)
    warmup_buffer(
        ts.replay,
        ts.encoder,
        grid_size=args.size,
        max_steps=args.max_steps,
        wall_length=args.wall_length,
        n_steps=args.warmup_steps,
        seed=args.seed,
        policy="random",
    )
    print(f"done ({len(ts.replay)} transitions)\n")

    # --- Train ---------------------------------------------------------------
    result = train_phase(
        ts,
        grid_size=args.size,
        max_steps=args.max_steps,
        wall_length=args.wall_length,
        n_episodes=args.episodes,
        use_reward_shaping=not args.no_reward_shaping,
        shaping_weight=args.shaping_weight,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_episodes=args.eps_decay_episodes,
        gamma=args.gamma,
        batch_size=args.batch_size,
        target_update_steps=args.target_update_steps,
        grad_clip=args.grad_clip,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        save_interval=args.save_interval,
        save_dir=args.save_dir,
        render_eval=args.render_eval,
    )

    # --- Final prints + final eval ------------------------------------------
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Episodes:     {result['episodes']}")
    print(f"  Global steps: {result['global_step']:,}")
    print(f"  Best SR:      {result['global_best_sr']*100:.1f}%")
    print(f"  Time:         {result['elapsed']:.1f}s ({result['elapsed']/60:.1f} min)")

    print("\n  ── Final Evaluation (50 episodes, greedy) ──")
    final = evaluate(
        q_net,
        encoder,
        device,
        n_episodes=50,
        grid_size=args.size,
        max_steps=args.max_steps,
        wall_length=args.wall_length,
    )
    print(
        f"  Success={final['success_rate']*100:.1f}% │ "
        f"AvgSteps={final['avg_steps']:.1f} │ "
        f"Range=[{final['min_steps']}, {final['max_steps']}]"
    )

    save_checkpoint(
        os.path.join(args.save_dir, "final_model.pt"),
        episode=result["episodes"],
        q_net=q_net,
        target_net=target_net,
        optimizer=optimizer,
        epsilon=args.eps_end,
        global_step=ts.global_step,
        best_success_rate=ts.best_success_rate,
    )
    print("  [Final model → final_model.pt]\nDone! 🎯")


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum runner
# ─────────────────────────────────────────────────────────────────────────────

def train_curriculum(args: argparse.Namespace) -> None:
    """Multi-phase curriculum training with progressive difficulty."""

    from curriculum import default_curriculum, aggressive_curriculum, PhaseTracker

    # --- Reproducibility seeds ----------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    encoder = StateEncoder(args.size)

    q_net = QNetwork(encoder.feature_dim, 4).to(device)
    target_net = QNetwork(encoder.feature_dim, 4).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.replay_size)

    if args.resume and os.path.exists(args.resume):
        _ = load_checkpoint(args.resume, q_net, target_net, optimizer, device)
        print(f"  Resumed weights from {args.resume}")

    ts = TrainState(
        q_net=q_net,
        target_net=target_net,
        optimizer=optimizer,
        replay=replay,
        encoder=encoder,
        device=device,
    )

    curriculum = aggressive_curriculum() if args.preset == "aggressive" else default_curriculum()
    curriculum.eval_episodes = args.eval_episodes
    curriculum.eval_interval = args.eval_interval

    param_count = sum(p.numel() for p in q_net.parameters())
    print("=" * 70)
    print("  DQN TRAINING — GridWorld Navigation (CURRICULUM)")
    print("=" * 70)
    print(f"  Device:            {device}")
    print(f"  State dim:         {encoder.feature_dim}")
    print(f"  Network params:    {param_count:,}")
    print(f"  Grid:              {args.size}×{args.size}")
    print(f"  Preset:            {args.preset}")
    print(f"  Reward shaping:    {'ON (per-phase weight)' if not args.no_reward_shaping else 'OFF'}")
    print(f"  Learning rate:     {args.lr}")
    print(f"  Double DQN:        ON")
    print(f"  Phases:            {len(curriculum.phases)}")
    print(f"  Max total eps:     {curriculum.total_max_episodes}")
    print()
    print(curriculum.summary())
    print("=" * 70 + "\n")

    total_start = time.time()
    total_episodes = 0

    for phase_idx, phase in enumerate(curriculum.phases, 1):
        # Fresh replay each phase tends to help stability.
        ts.replay = ReplayBuffer(args.replay_size)

        # More warmup in later phases (harder env).
        warm_steps = args.warmup_steps if phase_idx == 1 else args.warmup_steps * 5

        warmup_buffer(
            ts.replay,
            ts.encoder,
            grid_size=args.size,
            max_steps=args.max_steps,
            wall_length=phase.wall_length,
            n_steps=warm_steps,
            seed=args.seed + phase_idx,
            policy="bfs",
            eps=0.05,
        )

        tracker = PhaseTracker(phase=phase)

        print("─" * 70)
        print(f"  PHASE {phase_idx}/{len(curriculum.phases)}: {phase.name}")
        print(
            f"  wall_length={phase.wall_length}  │  "
            f"max_eps={phase.max_episodes}  │  "
            f"advance@{phase.advance_threshold*100:.0f}% "
            f"(×{phase.consecutive_required})  │  "
            f"ε={phase.eps_reset:.2f}  │  "
            f"shaping={phase.shaping_weight:.1f}"
        )
        print("─" * 70 + "\n")

        # Epsilon decays over most of the phase.
        phase_decay = int(phase.max_episodes * 0.85)

        def on_eval(sr: float) -> bool:
            # Advance when threshold is met consecutive times.
            return tracker.record_eval(sr)

        result = train_phase(
            ts,
            grid_size=args.size,
            max_steps=args.max_steps,
            wall_length=phase.wall_length,
            n_episodes=phase.max_episodes,
            use_reward_shaping=not args.no_reward_shaping,
            shaping_weight=phase.shaping_weight,
            eps_start=phase.eps_reset,
            eps_end=args.eps_end,
            eps_decay_episodes=phase_decay,
            gamma=args.gamma,
            batch_size=args.batch_size,
            target_update_steps=args.target_update_steps,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
            eval_interval=curriculum.eval_interval,
            eval_episodes=curriculum.eval_episodes,
            save_interval=args.save_interval,
            save_dir=args.save_dir,
            render_eval=args.render_eval,
            phase_name=phase.name,
            on_eval=on_eval,
        )

        total_episodes += result["episodes"]
        advanced_early = result["episodes"] < phase.max_episodes
        reason = "threshold met" if advanced_early else "max episodes reached"

        print(f"  Phase '{phase.name}' complete ({reason})")
        print(
            f"  Episodes: {result['episodes']}  │  "
            f"Phase best SR: {result['phase_best_sr']*100:.1f}%  │  "
            f"Global best SR: {result['global_best_sr']*100:.1f}%\n"
        )

    target_wall = curriculum.phases[-1].wall_length
    total_elapsed = time.time() - total_start

    print("=" * 70)
    print("  CURRICULUM TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total episodes:    {total_episodes}")
    print(f"  Total steps:       {ts.global_step:,}")
    print(f"  Total time:        {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Global best SR:    {ts.best_success_rate*100:.1f}%")

    print(f"\n  ── Final Evaluation (50 episodes, wall_length={target_wall}, greedy) ──")
    final = evaluate(
        q_net,
        encoder,
        device,
        n_episodes=50,
        grid_size=args.size,
        max_steps=args.max_steps,
        wall_length=target_wall,
    )
    print(
        f"  Success={final['success_rate']*100:.1f}% │ "
        f"AvgSteps={final['avg_steps']:.1f} │ "
        f"Range=[{final['min_steps']}, {final['max_steps']}]"
    )

    save_checkpoint(
        os.path.join(args.save_dir, "final_curriculum_model.pt"),
        episode=total_episodes,
        q_net=q_net,
        target_net=target_net,
        optimizer=optimizer,
        epsilon=args.eps_end,
        global_step=ts.global_step,
        best_success_rate=ts.best_success_rate,
        extra={"curriculum_preset": args.preset},
    )
    print("  [Final model → final_curriculum_model.pt]\nDone! 🎯")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DQN Training for GridWorld Navigation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                                    # Single-phase (defaults)
  python train.py --curriculum                       # Curriculum learning
  python train.py --curriculum --preset aggressive   # Faster curriculum
  python train.py --episodes 20000                   # More episodes (single)
  python train.py --no-reward-shaping                # Sparse reward only
  python train.py --resume checkpoints/best_model.pt # Resume training
  python train.py --render-eval                      # Visualize evals
""",
    )

    # Mode
    mode = p.add_argument_group("Training Mode")
    mode.add_argument("--curriculum", action="store_true", help="Enable curriculum learning")
    mode.add_argument(
        "--preset",
        type=str,
        default="default",
        choices=["default", "aggressive"],
        help="Curriculum preset (default: default)",
    )

    # Environment
    env = p.add_argument_group("Environment")
    env.add_argument("--size", type=int, default=ENV_CONFIG["size"], help="Grid size including borders")
    env.add_argument("--max-steps", type=int, default=ENV_CONFIG["max_steps"], help="Max steps per episode")
    env.add_argument(
        "--wall-length",
        type=int,
        default=ENV_CONFIG["wall_length"],
        help="Wall length for single-phase",
    )

    # DQN hyperparameters
    dqn = p.add_argument_group("DQN Hyperparameters")
    dqn.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    dqn.add_argument("--gamma", type=float, default=AGENT_CONFIG["discount_factor"], help="Discount factor")
    dqn.add_argument("--batch-size", type=int, default=256, help="Replay batch size")
    dqn.add_argument("--replay-size", type=int, default=100_000, help="Buffer capacity")
    dqn.add_argument("--warmup-steps", type=int, default=5000, help="Warmup steps")
    dqn.add_argument("--target-update-steps", type=int, default=1000, help="Steps between target updates")
    dqn.add_argument("--grad-clip", type=float, default=10.0, help="Max gradient norm")

    # Exploration
    exp = p.add_argument_group("Exploration")
    exp.add_argument("--eps-start", type=float, default=1.0, help="Initial epsilon")
    exp.add_argument("--eps-end", type=float, default=0.05, help="Final epsilon")
    exp.add_argument("--eps-decay-episodes", type=int, default=12_000, help="Eps decay episodes")

    # Training
    trn = p.add_argument_group("Training")
    trn.add_argument("--episodes", type=int, default=TRAIN_CONFIG["n_episodes"], help="Training episodes")
    trn.add_argument("--no-reward-shaping", action="store_true", help="Disable reward shaping")
    trn.add_argument("--shaping-weight", type=float, default=0.3, help="Distance shaping weight")
    trn.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")

    # Logging & saving
    log = p.add_argument_group("Logging & Saving")
    log.add_argument("--log-interval", type=int, default=50)
    log.add_argument("--eval-interval", type=int, default=500)
    log.add_argument("--eval-episodes", type=int, default=25)
    log.add_argument("--save-interval", type=int, default=TRAIN_CONFIG["save_interval"])
    log.add_argument("--save-dir", type=str, default=PATHS["save_dir"])
    log.add_argument("--render-eval", action="store_true")

    # Misc
    misc = p.add_argument_group("Misc")
    misc.add_argument("--seed", type=int, default=42)
    misc.add_argument("--cpu", action="store_true", help="Force CPU")

    return p.parse_args()


if __name__ == "__main__":
    args = build_args()
    if args.curriculum:
        train_curriculum(args)
    else:
        train_single(args)
