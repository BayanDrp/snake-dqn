import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


csv_path = sys.argv[1] if len(sys.argv) > 1 else "logs/ppo_training_log.csv"
out_path = sys.argv[2] if len(sys.argv) > 2 else "img/ppo_learning_curve.png"

if not os.path.exists(csv_path):
    raise SystemExit(f"Log not found: {csv_path}")

with open(csv_path, newline="") as file:
    rows = list(csv.DictReader(file))

episodes = [int(row["episode"]) for row in rows]
rewards = [float(row["reward"]) for row in rows]
best = [float(row["best"]) for row in rows]

os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
plt.plot(episodes, rewards, alpha=0.4, label="reward")
plt.plot(episodes, best, label="best reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("PPO Learning Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(out_path)
print(f"Saved {out_path}")
