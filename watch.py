"""Watch a trained agent play Snake with the PyGame GUI (no training).

Usage:
    python watch.py                        # uses best_snake.pth
    python watch.py --model my_model.pth
    python watch.py --grid-size 12 --vision-radius 3 --fps 15
    python watch.py --episodes 3
"""

import os
import sys

# Headless (Colab/CI): keep PyGame from crashing on display.open.
if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    os.environ["SDL_VIDEODRIVER"] = "dummy"

import torch

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from src.env import SnakeEnv
from src.render import SnakeGame
from agent.agent import Agent


def parse_args(argv):
    args = {
        "model": "best_snake.pth",
        "grid_size": 12,
        "vision_radius": 3,
        "fps": 15,
        "episodes": 1,
        "max_steps": 500,
    }
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--model", "-m") and i + 1 < len(argv):
            args["model"] = argv[i + 1]; i += 2
        elif a in ("--grid-size", "-g") and i + 1 < len(argv):
            args["grid_size"] = int(argv[i + 1]); i += 2
        elif a in ("--vision-radius", "-v") and i + 1 < len(argv):
            args["vision_radius"] = int(argv[i + 1]); i += 2
        elif a == "--fps" and i + 1 < len(argv):
            args["fps"] = int(argv[i + 1]); i += 2
        elif a in ("--episodes", "-e") and i + 1 < len(argv):
            args["episodes"] = int(argv[i + 1]); i += 2
        elif a in ("--max-steps", "-s") and i + 1 < len(argv):
            args["max_steps"] = int(argv[i + 1]); i += 2
        elif a in ("--help", "-h"):
            print(__doc__)
            sys.exit(0)
        else:
            i += 1
    return args


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if not os.path.exists(args["model"]):
        print(f"Model not found: {args['model']}")
        print("Train one first: python train.py --save best_snake.pth")
        sys.exit(1)

    env = SnakeEnv(grid_size=args["grid_size"], vision_radius=args["vision_radius"])
    agent = Agent(env, vision_radius=args["vision_radius"], max_steps=args["max_steps"])

    # Load trained weights into BOTH networks
    agent.model.load_state_dict(torch.load(args["model"], map_location="cpu"))
    agent.target.load_state_dict(agent.model.state_dict())

    renderer = SnakeGame(grid_size=args["grid_size"], vision_radius=args["vision_radius"])
    renderer.env = env

    for ep in range(1, args["episodes"] + 1):
        state = env.reset()
        done = False
        steps = 0
        total_reward = 0.0

        while not done and steps < args["max_steps"]:
            # Handle window close
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Closed.")
                    sys.exit(0)

            action = agent.model.select_action(state, 0.0, agent.device)
            state, reward, done = env.step(action)
            total_reward += reward

            renderer.render()
            renderer.clock.tick(args["fps"])
            steps += 1

        print(f"Episode {ep}/{args['episodes']}  score={env.score}  "
              f"steps={steps}  reward={total_reward:.2f}")

    print(f"Done. Played {args['episodes']} episode(s).")


if __name__ == "__main__":
    main()