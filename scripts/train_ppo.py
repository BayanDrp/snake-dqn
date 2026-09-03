import os
import sys
import torch
import csv

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

from game.env import SnakeEnv
from ppo.agent import PPOAgent


env = SnakeEnv(grid_size=12, vision_radius=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
agent = PPOAgent(state_dim=50, action_dim=3, device=device)
os.makedirs("logs", exist_ok=True)
best = float("-inf")

with open("logs/ppo_training_log.csv", "w", newline="") as file:
    log = csv.writer(file)
    log.writerow(["episode", "reward", "best"])

    for episode in range(1000):
        reward = agent.train_episode(env, max_steps=500)
        best = max(best, reward)
        log.writerow([episode + 1, reward, best])
        file.flush()
        print(f"Episode {episode + 1}: reward = {reward:.2f}")

torch.save(
    {
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
    },
    "ppo_snake.pth",
)
