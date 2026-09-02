#!/usr/bin/env bash
# Render a trained (or random) agent to img/<name>.gif, headless.
#
# Usage:
#   ./scripts/train_for_gif.sh --model best_snake.pth --out img/ep5000.gif [--episodes 1]
#
# If --model is omitted, the agent plays randomly (useful for the "before" GIF).

set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SRC_DIR"

MODEL=""
OUT="img/episodes.gif"
EPISODES=1
FPS=12
GRID=12
RADIUS=3
FRAMES_DIR="$(mktemp -d)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)  MODEL="$2";  shift 2 ;;
    --out)    OUT="$2";    shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --fps)    FPS="$2";    shift 2 ;;
    --grid-size) GRID="$2"; shift 2 ;;
    --vision-radius) RADIUS="$2"; shift 2 ;;
    --frames) FRAMES_DIR="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

if [[ "$MODEL" != "" && ! -f "$MODEL" ]]; then
  echo "model not found: $MODEL"
  exit 1
fi

mkdir -p "$(dirname "$OUT")"

PY_MODEL="$MODEL" PY_OUT="$FRAMES_DIR" PY_EPISODES="$EPISODES" \
PY_GRID="$GRID" PY_RADIUS="$RADIUS" \
SDL_VIDEODRIVER=dummy python3 - <<'EOF'
import os, sys
sys.path.insert(0, os.path.abspath("."))

import pygame
import torch

from game.env import SnakeEnv
from game.render import SnakeGame
from dqn.agent import Agent

model = os.environ.get("PY_MODEL", "")
out = os.environ["PY_OUT"]
episodes = int(os.environ["PY_EPISODES"])
grid = int(os.environ["PY_GRID"])
radius = int(os.environ["PY_RADIUS"])

env = SnakeEnv(grid_size=grid, vision_radius=radius)
ag = Agent(env, vision_radius=radius, max_steps=500)

if model:
    ag.model.load_state_dict(
        torch.load(model, map_location="cpu", weights_only=True)
    )

game = SnakeGame(grid_size=grid, vision_radius=radius)
game.env = env

frame = 0
for ep in range(episodes):
    state = env.reset()
    done = False
    steps = 0
    while not done and steps < 500:
        eps = 0.0 if model else 1.0   # trained: greedy; no model: random
        action = ag.model.select_action(state, eps, ag.device)
        state, reward, done = env.step(action)
        game.render()
        path = os.path.join(out, f"f{frame:05d}.png")
        pygame.image.save(game.screen, path)
        frame += 1
        steps += 1

print(f"captured {frame} frames")
EOF

echo "assembling GIF with ffmpeg -> $OUT"
ffmpeg -y -framerate "$FPS" -i "$FRAMES_DIR/f%05d.png" \
  -vf "format=rgb24" -loop 0 "$OUT"

echo "cleanup: removing $FRAMES_DIR"
rm -rf "$FRAMES_DIR"
echo "done: $OUT"