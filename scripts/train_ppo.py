import os
import sys
import torch
import csv

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

from game.env import SnakeEnv
from ppo.agent import PPOAgent


def parse_args(argv):
    args = {
        "episodes": 1000,
        "max_steps": 500,
        "save": "ppo_snake.pth",
        "log": "logs/ppo_training_log.csv",
    }
    i = 0
    while i < len(argv):
        if argv[i] in ("--episodes", "-e"):
            args["episodes"] = int(argv[i + 1]); i += 2
        elif argv[i] in ("--max-steps", "-m"):
            args["max_steps"] = int(argv[i + 1]); i += 2
        elif argv[i] in ("--save", "-s"):
            args["save"] = argv[i + 1]; i += 2
        elif argv[i] == "--log":
            args["log"] = argv[i + 1]; i += 2
        else:
            i += 1
    return args


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    env = SnakeEnv(grid_size=12, vision_radius=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(
        state_dim=53,
        action_dim=3,
        device=device,
        entropy_coef=0.02,
    )
    print(f"Using device: {device}")

    os.makedirs(os.path.dirname(args["log"]) or ".", exist_ok=True)
    best = float("-inf")
    with open(args["log"], "w", newline="") as file:
        log = csv.writer(file)
        log.writerow(["episode", "reward", "best"])
        for episode in range(1, args["episodes"] + 1):
            reward = agent.train_episode(env, args["max_steps"])
            best = max(best, reward)
            log.writerow([episode, reward, best])
            print(
                f"Episode {episode}/{args['episodes']}: "
                f"reward={reward:.2f}, score={env.score}"
            )

    torch.save({
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
    }, args["save"])
    print(f"Model saved to {args['save']}")


if __name__ == "__main__":
    main()
