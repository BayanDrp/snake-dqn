"""Training entry point: train the DQN to play Snake, with optional live rendering.

Usage:
    python train.py                       # train headless (fast)
    python train.py --episodes 3000       # more episodes
    python train.py --render-every 100    # pop a PyGame window every 100 episodes
    python train.py --save best.pth       # save the trained model
"""

import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "x11")

import torch

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from src.env import SnakeEnv
from agent.agent import Agent


def parse_args(argv):
    args = {
        "grid_size": 12,
        "vision_radius": 3,
        "episodes": 1000,
        "render_every": 10,      # 0 = never render (headless)
        "fps": 15,
        "save": "best_snake.pth",
        "log_every": 50,
        "max_steps": 500,
    }
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--grid-size", "-g") and i + 1 < len(argv):
            args["grid_size"] = int(argv[i + 1]); i += 2
        elif a in ("--vision-radius", "-v") and i + 1 < len(argv):
            args["vision_radius"] = int(argv[i + 1]); i += 2
        elif a in ("--episodes", "-e") and i + 1 < len(argv):
            args["episodes"] = int(argv[i + 1]); i += 2
        elif a in ("--render-every", "-r") and i + 1 < len(argv):
            args["render_every"] = int(argv[i + 1]); i += 2
        elif a == "--fps" and i + 1 < len(argv):
            args["fps"] = int(argv[i + 1]); i += 2
        elif a in ("--save", "-s") and i + 1 < len(argv):
            args["save"] = argv[i + 1]; i += 2
        elif a == "--log-every" and i + 1 < len(argv):
            args["log_every"] = int(argv[i + 1]); i += 2
        elif a in ("--max-steps", "-m") and i + 1 < len(argv):
            args["max_steps"] = int(argv[i + 1]); i += 2
        elif a in ("--help", "-h"):
            print(__doc__)
            sys.exit(0)
        else:
            i += 1
    return args


def make_renderer(grid_size, vision_radius):
    """Build a PyGame renderer lazily (creating a window is expensive)."""
    from src.render import SnakeGame
    return SnakeGame(grid_size=grid_size, vision_radius=vision_radius)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    print(f"grid={args['grid_size']} vision_radius={args['vision_radius']} "
          f"episodes={args['episodes']} max_steps={args['max_steps']}")

    env = SnakeEnv(grid_size=args["grid_size"], vision_radius=args["vision_radius"])
    agent = Agent(
        env,
        vision_radius=args["vision_radius"],
        max_steps=args["max_steps"],
    )

    renderer = None
    if args["render_every"]:
        renderer = make_renderer(args["grid_size"], args["vision_radius"])

    best_reward = float("-inf")
    start = time.time()

    for episode in range(1, args["episodes"] + 1):
        total_reward = agent.train_episode()

        if total_reward > best_reward:
            best_reward = total_reward

        # Periodically render the CURRENT model playing greedily
        if renderer and episode % args["render_every"] == 0:
            renderer.env = env            # point renderer at the same real env
            renderer.done = False
            state = env.reset()
            done = False
            steps = 0
            while not done and steps < args["max_steps"]:
                action = agent.model.select_action(state, 0.0, agent.device)
                state, reward, done = env.step(action)
                renderer.render()
                renderer.clock.tick(args["fps"])
                steps += 1

        if episode % args["log_every"] == 0 or episode == 1:
            el = time.time() - start
            avg = best_reward
            print(f"Episode {episode}/{args['episodes']}  "
                  f"reward={total_reward:7.2f}  best={best_reward:7.2f}  "
                  f"eps={agent.epsilon():.3f}  "
                  f"buffer={len(agent.replay_buffer)}  "
                  f"elapsed={el:.1f}s", flush=True)

    # Final metric: greedy evaluation
    avg_reward = agent.evaluate(10)
    print(f"\nFinal greedy eval over 10 episodes: avg reward {avg_reward:.2f}")

    if args["save"]:
        torch.save(agent.model.state_dict(), args["save"])
        print(f"Model saved to {args['save']}")


if __name__ == "__main__":
    main()