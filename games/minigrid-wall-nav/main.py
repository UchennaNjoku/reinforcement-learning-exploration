"""
Main entry point for MiniGrid Wall Navigation.
==============================================

Unified CLI for training and evaluating Q-learning agents on the wall navigation task.

Commands:
    train   - Train a Q-learning agent (default: CustomAgent11)
    eval    - Evaluate a trained agent
    manual  - Play manually with keyboard

Usage:
    python main.py train                   # Train with defaults
    python main.py train --agent CustomAgent13 --episodes 20000 --curriculum
    python main.py eval                    # Evaluate best checkpoint
    python main.py eval --checkpoint checkpoints/q_table_ep15000.pkl
    python main.py manual                  # Play manually
"""

from __future__ import annotations

import argparse
import sys

from train import train_agent, AGENT_CLASSES
from evaluate import evaluate_agent, manual_control


def main():
    parser = argparse.ArgumentParser(
        description="Distance-Aware Tabular Q-Learning on MiniGrid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train                    # Train with default settings
  python main.py train --episodes 5000    # Train for 5000 episodes
  python main.py eval                     # Evaluate trained agent
  python main.py eval --episodes 20       # Evaluate for 20 episodes
  python main.py manual                   # Play manually
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the agent")
    train_parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of training episodes",
    )
    train_parser.add_argument(
        "--render",
        action="store_true",
        help="Render training (slow)",
    )
    train_parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save Q-table",
    )
    train_parser.add_argument(
        "--agent",
        type=str,
        default="CustomAgent11",
        choices=["CustomAgent", "CustomAgent2", "CustomAgent3", "CustomAgent4", "CustomAgent5", "CustomAgent6", "CustomAgent7", "CustomAgent8", "CustomAgent9", "CustomAgent10", "CustomAgent11", "CustomAgent12", "CustomAgent13"],
        help="Which agent class to use (default: CustomAgent11)",
    )
    train_parser.add_argument(
        "--epsilon-delay",
        type=int,
        default=None,
        dest="epsilon_delay",
        help="Number of episodes to delay epsilon decay (default: 1350)",
    )
    train_parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=None,
        dest="epsilon_decay",
        help="Epsilon decay rate per episode (default: 0.9995)",
    )
    train_parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable curriculum learning (progressive wall difficulty)",
    )

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate the agent")
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to Q-table checkpoint",
    )
    eval_parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of evaluation episodes",
    )
    eval_parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering",
    )
    eval_parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between steps",
    )

    # Manual command
    manual_parser = subparsers.add_parser("manual", help="Manual control mode")

    args = parser.parse_args()

    if args.command == "train":
        from train import AGENT_CLASSES
        train_agent(
            n_episodes=args.episodes,
            render=args.render,
            save_path=args.save_path,
            agent_class=AGENT_CLASSES.get(args.agent),
            epsilon_delay=args.epsilon_delay,
            epsilon_decay=args.epsilon_decay,
            curriculum=args.curriculum,
        )
    elif args.command == "eval":
        evaluate_agent(
            checkpoint_path=args.checkpoint,
            n_episodes=args.episodes,
            render=not args.no_render,
            delay=args.delay,
        )
    elif args.command == "manual":
        manual_control()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
