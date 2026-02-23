"""
Evaluation Script for GridWorld Navigation
==========================================

Load a trained checkpoint and watch the agent play, or run statistics without rendering.

Usage:
    python eval.py                              # Watch best_model.pt
    python eval.py --checkpoint checkpoints/final_curriculum_model.pt
    python eval.py --episodes 5 --delay 0.2     # Slower, fewer episodes
    python eval.py --no-render --episodes 100   # Just stats, no visualization
    python eval.py --sweep                      # Test across all wall lengths
"""

from __future__ import annotations

import argparse
import os
import time
import random

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("PyTorch is required. Install with: pip install torch")
    exit(1)

from env import GridWorldEnv
from config import ENV_CONFIG, PATHS
from train import StateEncoder, QNetwork


# ─────────────────────────────────────────────────────────────────────────────
# Load Checkpoint
# ─────────────────────────────────────────────────────────────────────────────
def load_checkpoint(filepath: str, q_net: nn.Module, device: torch.device):
    """Load model weights from checkpoint."""
    ckpt = torch.load(filepath, map_location=device, weights_only=True)
    q_net.load_state_dict(ckpt["q_state_dict"])
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    q_net: nn.Module,
    encoder: StateEncoder,
    device: torch.device,
    n_episodes: int = 10,
    grid_size: int = 12,
    max_steps: int = 120,
    wall_length: int = 4,
    render: bool = True,
    delay: float = 0.1,
    seed: int | None = None,
) -> dict:
    """
    Run evaluation episodes with greedy policy.

    Returns dict with success_rate, avg_steps, min_steps, max_steps, total_steps.
    """
    env = GridWorldEnv(
        size=grid_size,
        max_steps=max_steps,
        wall_length=wall_length,
        render_mode="human" if render else None,
    )

    q_net.eval()

    successes = 0
    steps_list = []
    episode_rewards = []

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    try:
        for ep in range(1, n_episodes + 1):
            obs, info = env.reset(seed=(seed + ep) if seed is not None else None)
            done = False
            steps = 0
            episode_reward = 0.0
            terminated = False

            if render:
                print(f"\n--- Episode {ep}/{n_episodes} ---")
                print(env._render_text())
                time.sleep(delay)

            while not done:
                state = encoder.encode_tensor(obs, env, device)
                q_values = q_net(state)
                action = int(q_values.argmax(dim=1).item())

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
                episode_reward += reward

                if terminated:
                    successes += 1

                if render:
                    print(f"  Step {steps}: action={['UP','DOWN','LEFT','RIGHT'][action]}, "
                          f"reward={reward:.2f}")
                    print(env._render_text())
                    time.sleep(delay)

            steps_list.append(steps)
            episode_rewards.append(episode_reward)

            if render:
                status = "✓ SUCCESS" if terminated else "✗ FAILED"
                print(f"  Result: {status} in {steps} steps (reward: {episode_reward:.2f})")
                time.sleep(delay * 2)

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")

    finally:
        env.close()

    total_episodes = len(steps_list)
    if total_episodes == 0:
        return {
            "success_rate": 0.0,
            "avg_steps": 0.0,
            "min_steps": 0,
            "max_steps": 0,
            "total_steps": 0,
            "avg_reward": 0.0,
            "episodes_completed": 0,
        }

    return {
        "success_rate": successes / total_episodes,
        "avg_steps": float(np.mean(steps_list)),
        "min_steps": min(steps_list),
        "max_steps": max(steps_list),
        "total_steps": sum(steps_list),
        "avg_reward": float(np.mean(episode_rewards)),
        "episodes_completed": total_episodes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Wall-Length Sweep
# ─────────────────────────────────────────────────────────────────────────────
def run_sweep(
    q_net: nn.Module,
    encoder: StateEncoder,
    device: torch.device,
    wall_lengths: list[int],
    n_episodes: int = 50,
    grid_size: int = 12,
    max_steps: int = 120,
    seed: int | None = None,
):
    """
    Evaluate the same checkpoint across multiple wall lengths.
    Useful after curriculum training to verify the agent didn't
    forget easier configurations.
    """
    print("\n" + "=" * 60)
    print("  WALL-LENGTH SWEEP")
    print("=" * 60)
    print(f"  Episodes per wall length: {n_episodes}")
    print(f"  Wall lengths: {wall_lengths}")
    print("=" * 60 + "\n")

    results = {}
    for wl in wall_lengths:
        print(f"  Evaluating wall_length={wl}...", end=" ", flush=True)
        r = evaluate(
            q_net, encoder, device,
            n_episodes=n_episodes,
            grid_size=grid_size,
            max_steps=max_steps,
            wall_length=wl,
            render=False,
            seed=seed,
        )
        results[wl] = r
        print(f"Success={r['success_rate']*100:.1f}%  │  "
              f"AvgSteps={r['avg_steps']:.1f}  │  "
              f"Range=[{r['min_steps']}, {r['max_steps']}]")

    # Summary table
    print("\n" + "─" * 60)
    print(f"  {'Wall':>6s}  │  {'Success':>8s}  │  {'AvgSteps':>8s}  │  {'Min':>4s}  │  {'Max':>4s}")
    print("─" * 60)
    for wl in wall_lengths:
        r = results[wl]
        print(f"  {wl:>6d}  │  {r['success_rate']*100:>7.1f}%  │  "
              f"{r['avg_steps']:>8.1f}  │  {r['min_steps']:>4d}  │  {r['max_steps']:>4d}")
    print("─" * 60)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained GridWorld Navigation agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval.py                              # Watch best_model.pt
  python eval.py --checkpoint checkpoints/final_curriculum_model.pt
  python eval.py --episodes 5 --delay 0.2     # Slower, fewer episodes
  python eval.py --no-render --episodes 100   # Just stats, no visualization
  python eval.py --sweep                      # Test across wall lengths 0-4
  python eval.py --sweep --sweep-lengths 0 2 4  # Custom wall lengths
        """,
    )

    # Checkpoint
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_model.pt",
        help="Path to checkpoint file (default: checkpoints/best_model.pt)",
    )

    # Evaluation settings
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Number of episodes to run (default: 10)",
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="Run without visualization (just print stats)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.1,
        help="Delay between steps in seconds when rendering (default: 0.1)",
    )

    # Sweep mode
    sweep_group = parser.add_argument_group("Sweep Mode")
    sweep_group.add_argument(
        "--sweep", action="store_true",
        help="Evaluate across multiple wall lengths",
    )
    sweep_group.add_argument(
        "--sweep-lengths", type=int, nargs="+", default=[0, 1, 2, 3, 4],
        help="Wall lengths to test in sweep mode (default: 0 1 2 3 4)",
    )
    sweep_group.add_argument(
        "--sweep-episodes", type=int, default=50,
        help="Episodes per wall length in sweep mode (default: 50)",
    )

    # Environment settings
    env_group = parser.add_argument_group("Environment")
    env_group.add_argument(
        "--size", type=int, default=ENV_CONFIG["size"],
        help=f"Grid size (default: {ENV_CONFIG['size']})",
    )
    env_group.add_argument(
        "--max-steps", type=int, default=ENV_CONFIG["max_steps"],
        help=f"Max steps per episode (default: {ENV_CONFIG['max_steps']})",
    )
    env_group.add_argument(
        "--wall-length", type=int, default=ENV_CONFIG["wall_length"],
        help=f"Wall length for standard eval (default: {ENV_CONFIG['wall_length']})",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")

    args = parser.parse_args()

    # Validate checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        save_dir = PATHS["save_dir"]
        if os.path.exists(save_dir):
            pts = sorted(f for f in os.listdir(save_dir) if f.endswith(".pt"))
            if pts:
                print(f"Available checkpoints in {save_dir}/:")
                for f in pts:
                    print(f"  - {save_dir}/{f}")
        exit(1)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    encoder = StateEncoder(args.size)
    q_net = QNetwork(encoder.feature_dim, 4).to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint, q_net, device)

    # Print checkpoint info
    print("\n" + "=" * 60)
    print("  GRIDWORLD NAVIGATION — EVALUATION")
    print("=" * 60)
    print(f"  Checkpoint:        {args.checkpoint}")
    if "episode" in ckpt:
        print(f"  Trained episodes:  {ckpt['episode']}")
    if "global_step" in ckpt:
        print(f"  Global steps:      {ckpt['global_step']:,}")
    if "best_success_rate" in ckpt:
        print(f"  Best success rate: {ckpt['best_success_rate']*100:.1f}%")
    if "phase" in ckpt:
        print(f"  Phase:             {ckpt['phase']}")
    if "wall_length" in ckpt:
        print(f"  Trained wall len:  {ckpt['wall_length']}")
    print(f"  Device:            {device}")
    print(f"  State dim:         {encoder.feature_dim}")
    print(f"  Grid size:         {args.size}×{args.size}")
    print(f"  Max steps:         {args.max_steps}")
    if not args.sweep:
        print(f"  Wall length:       {args.wall_length}")
        print(f"  Episodes:          {args.episodes}")
        print(f"  Render:            {'OFF' if args.no_render else 'ON'}")
        if not args.no_render:
            print(f"  Step delay:        {args.delay}s")
    else:
        print(f"  Mode:              SWEEP")
        print(f"  Wall lengths:      {args.sweep_lengths}")
        print(f"  Episodes/length:   {args.sweep_episodes}")
    print("=" * 60)

    # ── Sweep mode ──
    if args.sweep:
        run_sweep(
            q_net, encoder, device,
            wall_lengths=args.sweep_lengths,
            n_episodes=args.sweep_episodes,
            grid_size=args.size,
            max_steps=args.max_steps,
            seed=args.seed,
        )
        return

    # ── Standard evaluation ──
    print("\nStarting evaluation...\n")

    results = evaluate(
        q_net=q_net,
        encoder=encoder,
        device=device,
        n_episodes=args.episodes,
        grid_size=args.size,
        max_steps=args.max_steps,
        wall_length=args.wall_length,
        render=not args.no_render,
        delay=args.delay,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Episodes completed: {results['episodes_completed']}/{args.episodes}")
    print(f"  Success rate:       {results['success_rate']*100:.1f}%")
    print(f"  Average steps:      {results['avg_steps']:.1f}")
    print(f"  Min/Max steps:      {results['min_steps']} / {results['max_steps']}")
    print(f"  Total steps:        {results['total_steps']}")
    print(f"  Average reward:     {results['avg_reward']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()