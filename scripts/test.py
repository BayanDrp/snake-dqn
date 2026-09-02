"""Evaluate a trained agent headlessly and report performance metrics.

Usage:
    python scripts/test.py                         # tests best_snake.pth, 100 episodes
    python scripts/test.py --model my_model.pth
    python scripts/test.py --episodes 500 --grid-size 12 --vision-radius 3
    python scripts/test.py --render                # show live PyGame window while testing
"""

import os
import sys

# Headless (Colab/CI): keep PyGame from crashing on display.open.
if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    os.environ["SDL_VIDEODRIVER"] = "dummy"

import torch

src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from game.env import SnakeEnv
from dqn.agent import Agent


def parse_args(argv):
    args = {
        "model": "best_snake.pth",
        "grid_size": 12,
        "vision_radius": 3,
        "episodes": 100,
        "max_steps": 500,
        "fps": 15,
        "render": False,
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
        elif a in ("--episodes", "-e") and i + 1 < len(argv):
            args["episodes"] = int(argv[i + 1]); i += 2
        elif a in ("--max-steps", "-s") and i + 1 < len(argv):
            args["max_steps"] = int(argv[i + 1]); i += 2
        elif a == "--fps" and i + 1 < len(argv):
            args["fps"] = int(argv[i + 1]); i += 2
        elif a == "--render":
            args["render"] = True; i += 1
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
        print("Train one first: python scripts/train.py --save best_snake.pth")
        sys.exit(1)

    env = SnakeEnv(grid_size=args["grid_size"], vision_radius=args["vision_radius"])
    agent = Agent(env, vision_radius=args["vision_radius"], max_steps=args["max_steps"])
    agent.model.load_state_dict(torch.load(args["model"], map_location="cpu"))

    renderer = None
    if args["render"]:
        from game.render import SnakeGame
        import pygame
        renderer = SnakeGame(grid_size=args["grid_size"], vision_radius=args["vision_radius"])
        renderer.env = env

    scores, rewards, steps_list, outcomes = [], [], [], []
    max_score = 0

    for ep in range(1, args["episodes"] + 1):
        state = env.reset()
        done = False
        total = 0.0
        steps = 0

        while not done and steps < args["max_steps"]:
            if renderer:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit(0)

            action = agent.model.select_action(state, 0.0, agent.device)
            state, reward, done = env.step(action)
            total += reward
            steps += 1

            if renderer:
                renderer.render()
                renderer.clock.tick(args["fps"])

        scores.append(env.score)
        rewards.append(total)
        steps_list.append(steps)
        outcomes.append(done)  # False = hit max_steps (survived the cap)
        max_score = max(max_score, env.score)

    n = args["episodes"]
    avg_score = sum(scores) / n
    avg_reward = sum(rewards) / n
    avg_steps = sum(steps_list) / n
    best_score = max(scores)
    ate_food = sum(1 for s in scores if s > 0)
    never_crashed = sum(1 for o in outcomes if not o)

    print(f"\n=== Test report: {args['model']} over {n} episodes ===")
    print(f"  grid={args['grid_size']}  vision_radius={args['vision_radius']}  "
          f"max_steps={args['max_steps']}")
    print(f"  average score  : {avg_score:.2f}   (best: {best_score})")
    print(f"  average reward : {avg_reward:.2f}")
    print(f"  average steps  : {avg_steps:.1f}   ({args['max_steps']} = survived the cap)")
    print(f"  episodes eaten : {ate_food}/{n}  ({100.0 * ate_food / n:.1f}%)")
    print(f"  survived cap   : {never_crashed}/{n}  ({100.0 * never_crashed / n:.1f}%)")


if __name__ == "__main__":
    main()