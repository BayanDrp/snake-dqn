#!/usr/bin/env python3
"""Single entry point for every mode: train, watch, test, play.

Usage:
    python main.py train                       # train DQN (saves best_snake.pth)
    python main.py train --episodes 3000       # more episodes
    python main.py train --log logs/log.csv    # write CSV per episode

    python main.py train-ppo                   # train PPO (saves ppo_snake.pth)
    python main.py train-ppo --episodes 3000

    python main.py watch                       # watch best_snake.pth
    python main.py watch --model my.pth        # watch a specific model

    python main.py test                        # headless 100-game eval
    python main.py test --model my.pth

    python main.py play                        # you vs the snake (human mode)

All the actual work is in scripts/*.py — this just routes to them.
"""

import sys
import os

# Make sure game/ and dqn/ are importable from the project root.
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


def _has_display():
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


if not _has_display():
    os.environ["SDL_VIDEODRIVER"] = "dummy"

from scripts.train import main as train_main
from scripts.train_ppo import main as train_ppo_main
from scripts.watch import main as watch_main
from scripts.test import main as test_main
from scripts.play import main as play_main


COMMANDS = {
    "train":     train_main,
    "train-ppo": train_ppo_main,
    "watch":     watch_main,
    "test":      test_main,
    "play":      play_main,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__.strip())
        cmd = sys.argv[1] if len(sys.argv) > 1 else "(none)"
        print(f"\nUnknown command: {cmd!r}. Choose from: {', '.join(COMMANDS)}")
        sys.exit(1)

    command = sys.argv[1]
    # Drop the subcommand name so each script sees its own flags normally.
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    try:
        COMMANDS[command]()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
