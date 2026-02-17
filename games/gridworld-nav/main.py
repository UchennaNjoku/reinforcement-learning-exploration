"""
Main entry point for GridWorld Navigation.
============================================

Commands:
    manual  - Play manually with keyboard (default)
    test    - Run environment tests

Usage:
    python main.py              # Play manually (default)
    python main.py manual       # Play manually
    python main.py test         # Run tests
    python main.py --help       # Show help

"""

from __future__ import annotations

import argparse
import sys

from config import ENV_CONFIG


def manual_command(args):
    """Run manual control mode."""
    from play_manual import play_manual, play_text_mode
    
    if args.text:
        play_text_mode(
            size=args.size or ENV_CONFIG["size"],
            max_steps=args.max_steps or ENV_CONFIG["max_steps"],
            wall_length=args.wall_length or ENV_CONFIG["wall_length"],
        )
    else:
        play_manual(
            size=args.size or ENV_CONFIG["size"],
            max_steps=args.max_steps or ENV_CONFIG["max_steps"],
            wall_length=args.wall_length or ENV_CONFIG["wall_length"],
        )


def test_command(args):
    """Run test mode."""
    from test_env import test_environment, test_action_deltas
    
    if args.action_test:
        test_action_deltas()
    else:
        test_environment(
            episodes=args.episodes,
            render=args.render,
            delay=args.delay,
        )


def main():
    parser = argparse.ArgumentParser(
        description="GridWorld Navigation - Game 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Play manually with pygame
  python main.py --text             # Play in text mode
  python main.py test               # Run environment tests
  python main.py test --render      # Run tests with rendering
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Manual command (default)
    manual_parser = subparsers.add_parser("manual", help="Manual control mode")
    manual_parser.add_argument(
        "--size",
        type=int,
        default=None,
        help=f"Grid size (default: {ENV_CONFIG['size']})",
    )
    manual_parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=f"Max steps per episode (default: {ENV_CONFIG['max_steps']})",
    )
    manual_parser.add_argument(
        "--wall-length",
        type=int,
        default=None,
        help=f"Length of wall obstacle (default: {ENV_CONFIG['wall_length']})",
    )
    manual_parser.add_argument(
        "--text",
        action="store_true",
        help="Use text mode instead of graphical mode",
    )
    manual_parser.set_defaults(func=manual_command)
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of test episodes (default: 5)",
    )
    test_parser.add_argument(
        "--render",
        action="store_true",
        help="Render during tests",
    )
    test_parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between steps when rendering (default: 0.1)",
    )
    test_parser.add_argument(
        "--action-test",
        action="store_true",
        help="Only run action delta tests",
    )
    test_parser.set_defaults(func=test_command)
    
    # Parse args
    args = parser.parse_args()
    
    # Default to manual mode if no command specified
    if args.command is None:
        # Create manual args with defaults
        class Args:
            size = None
            max_steps = None
            wall_length = None
            text = False
            func = manual_command
        args = Args()
    
    # Run the selected command
    args.func(args)


if __name__ == "__main__":
    main()
