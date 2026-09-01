"""Play Snake yourself with the arrow keys (no learning). Uses the env + PyGame.

Actions: LEFT/RIGHT = turn the snake, the snake always moves forward.
Controls:
    LEFT  arrow  -> turn left
    RIGHT arrow  -> turn right
    Q / ESC      -> quit

Usage:
    python play.py
    python play.py --grid-size 15 --fps 10
"""

import os
import sys

# Headless (Colab/CI): keep PyGame from crashing on display.open.
if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from src.env import SnakeEnv
from src.render import SnakeGame

# In the env's convention actions are relative to the snake's direction:
#   0 = turn left, 1 = straight, 2 = turn right
KEY_TO_ACTION = {
    pygame.K_LEFT: 0,
    pygame.K_RIGHT: 2,
}


def parse_args(argv):
    args = {"grid_size": 12, "vision_radius": 3, "fps": 10}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--grid-size", "-g") and i + 1 < len(argv):
            args["grid_size"] = int(argv[i + 1]); i += 2
        elif a in ("--vision-radius", "-v") and i + 1 < len(argv):
            args["vision_radius"] = int(argv[i + 1]); i += 2
        elif a == "--fps" and i + 1 < len(argv):
            args["fps"] = int(argv[i + 1]); i += 2
        elif a in ("--help", "-h"):
            print(__doc__)
            sys.exit(0)
        else:
            i += 1
    return args


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])

    env = SnakeEnv(grid_size=args["grid_size"], vision_radius=args["vision_radius"])
    game = SnakeGame(grid_size=args["grid_size"], vision_radius=args["vision_radius"])
    game.env = env
    game.clock = pygame.time.Clock()

    state = env.reset()
    running = True
    paused = False

    while running:
        action = 1  # default: keep going straight

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key in KEY_TO_ACTION:
                    action = KEY_TO_ACTION[event.key]  # last key wins for this frame
                elif event.key == pygame.K_SPACE:
                    paused = not paused

        if not paused:
            state, reward, done = env.step(action)
            if done:
                print(f"Game over! Score: {env.score}  Steps: {env.steps}")

        game.render()
        game.clock.tick(args["fps"])

        if done:
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            state = env.reset()
                            done = False
                        elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                            running = False

    pygame.quit()


if __name__ == "__main__":
    main()