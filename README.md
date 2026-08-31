# Snake DQN — Watch My AI Learn

A from-scratch Deep Q-Network that learns to play Snake in the terminal. Built with pure NumPy — no frameworks.

## Demo

| Episode 1 (Random) | Episode 5000 (Trained) |
|:---:|:---:|
| ![ep1](img/ep1.gif) | ![ep5000](img/ep5000.gif) |

**Learning Curve**

![Learning Curve](img/learning_curve.png)

## Quick Start

```bash
# Install (only numpy required)
pip install -r requirements.txt

# Train the agent
python train.py

# Watch the trained agent play
python watch.py

# Play yourself (human mode)
python play.py
```

## How It Works

- **State**: Local vision window around snake head + current direction
- **Actions**: 3 relative moves — turn left, straight, turn right
- **Rewards**: +10 (eat), −10 (death), −0.01 per step (forces speed)
- **DQN**: Small MLP + Replay Buffer + Target Network + ε-greedy decay

## Structure

```
src/env.py       # Snake world (gym-style: reset/step)
src/render.py    # curses terminal renderer
agent/dqn.py     # MLP network
agent/memory.py  # Replay Buffer
agent/agent.py   # ε-greedy, target net sync
train.py         # Training loop
watch.py         # Watch trained agent
play.py          # Human playable
scripts/         # GIF generation + plotting
```

## Requirements

- Python 3.8+
- numpy (only required dependency)

## Credits

Built by [Drpisto](https://github.com/Drpisto)