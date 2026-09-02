# Snake DQN

A Snake game where a Deep Q-Network (DQN) learns to play by itself, built from scratch with **PyTorch** and a **PyGame** GUI. No RL framework — the whole DQN (replay buffer, target network, ε-greedy, Bellman update) is written from the ground up.

## Demo

| Before training (random) | After training (DQN) |
|:---:|:---:|
| ![ep1](img/ep1.gif) | ![ep5000](img/ep5000.gif) |

**Learning Curve** (real run: best reward climbs from −10 to +37 in 300 episodes)

![Learning Curve](img/learning_curve.png)

Verified DQN results on a trained model (100 games, greedy):

| Metric | Value |
|---|---|
| Average score | 3.87 |
| Best score | 25 |
| Games that ate food | 76 / 100 |
| Average reward | +32.6 |

## Quick Start

```bash
pip install -r requirements.txt

# Train the DQN agent (saves best_snake.pth)
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

### DQN (value-based) — implemented

Learn a Q-value per action → act greedily. Small MLP (vision + one-hot direction → Q-values), **replay buffer** (reuse old off-policy data), **target network** (stable Bellman targets), ε-greedy with linear decay.

### PPO (policy-gradient) — coming next

Plan: learn a direct policy π(a|s) → sample actions. **Actor-Critic** (shared trunk, actor head → action logits, critic head → V(s)), **GAE** (λ-weighted advantage estimation), **clipped objective** (limit update size), multi-epoch minibatch updates, entropy bonus for exploration. Scaffold lives in `ppo/` (empty, to be written).

## DQN vs PPO (the plan)

Same environment, same state/action/reward — two completely different learning families. The goal is to train both and compare learning curves head-to-head:

| | DQN | PPO |
|---|---|---|
| Learns | Q(s,a) values | policy π(a\|s) + value V(s) |
| Data use | Off-policy (replay buffer) | On-policy (fresh rollouts only) |
| Exploration | ε-greedy | stochastic policy sampling |
| Update rule | Bellman target + MSE | clipped surrogate objective |
| Sample efficiency | Better per episode (reuses data) | Worse (data used once) |
| Stability trick | Target network | Clip term |

Expectation on Snake: DQN usually reaches good scores faster per episode because replay reuses transitions; PPO (on-policy, no reuse) needs more episodes and often benefits from denser reward shaping on sparse-reward games like this one.

## Structure

```
game/env.py      # Snake world (gym-style: reset/step)
game/render.py   # PyGame GUI renderer
agent/           # DQN implementation
  dqn.py         #   DQN network (MLP) + ε-greedy action selection
  memory.py      #   Replay Buffer
  agent.py       #   DQN agent: online + target net, Bellman update, training
ppo/             # PPO scaffold — files empty, to be written
  __init__.py
  model.py       #   planned: Actor-Critic network (policy + value heads)
  buffer.py      #   planned: on-policy rollout buffer + GAE computation
  agent.py       #   planned: PPO trainer (clipped objective, entropy bonus)
  train.py       #   planned: PPO training entry (same --log CSV format)
scripts/
  train.py       # DQN training loop (--log CSV, --render-every)
  watch.py       # Watch trained agent
  test.py        # Headless evaluation with metrics report
  play.py        # Human playable
  make_plot.py   # Learning curve from logs/*
  compare.py     # planned: DQN vs PPO comparison plot
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