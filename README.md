# Snake DQN

A Snake game where a Deep Q-Network (DQN) learns to play by itself, built from scratch with **PyTorch** and a **PyGame** GUI. No RL framework — the DQN (replay buffer, target network, ε-greedy, Bellman update) is written from the ground up.

## Demo

| Before training (random) | After training (DQN) |
|:---:|:---:|
| ![ep1](img/ep1.gif) | ![ep5000](img/ep5000.gif) |

**Learning Curve** (real run: best reward climbs from −10 to +37 in 300 episodes)

![Learning Curve](img/learning_curve.png)

Verified results on a trained model (100 games, greedy):

| Metric | Value |
|---|---|
| Average score | 3.87 |
| Best score | 25 |
| Games that ate food | 76 / 100 |
| Average reward | +32.6 |

## Quick Start

```bash
pip install -r requirements.txt

# Train the agent (saves best_snake.pth)
python scripts/train.py --episodes 5000

# Test a trained model
python scripts/test.py

# Watch the trained agent play
python scripts/watch.py

# Play yourself (human mode)
python scripts/play.py
```

### Train on Google Colab (free GPU/CPU)

```python
!git clone https://github.com/BayanDrp/snake-dqn.git
%cd snake-dqn
!python scripts/train.py --episodes 5000 --save best_snake.pth --log logs/training_log.csv
!python scripts/test.py --model best_snake.pth --episodes 100
from google.colab import files
files.download('best_snake.pth')
files.download('logs/training_log.csv')
```

## How It Works

- **State**: local vision window around the snake head + current direction
- **Actions**: 3 relative moves — turn left, straight, turn right
- **Rewards**: +10 (eat), −10 (death), −0.01 per step (forces speed)
- **DQN**: small MLP (vision + one-hot direction → Q-values), replay buffer, target network, ε-greedy with linear decay

## Structure

```
game/env.py      # Snake world (gym-style: reset/step)
game/render.py   # PyGame GUI renderer
agent/dqn.py     # DQN network (MLP) + ε-greedy action selection
agent/memory.py  # Replay Buffer
agent/agent.py   # DQN agent: online + target net, Bellman update, training
scripts/
  train.py       # Training loop (--log CSV, --render-every)
  watch.py       # Watch trained agent
  test.py        # Headless evaluation with metrics report
  play.py        # Human playable
  make_plot.py   # Learning curve from logs/*
  train_for_gif.sh  # Demo GIF renderer
img/             # Demo GIFs + learning curve
logs/            # Training data (training_log.csv)
```

## Requirements

- Python 3.9+
- torch, numpy, pygame, matplotlib, pillow
- ffmpeg (only for generating the demo GIFs)

## Credits

Built by [Drpisto](https://github.com/Drpisto)