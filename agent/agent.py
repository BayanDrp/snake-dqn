import numpy as np
import torch
import torch.optim as optim

from .memory import ReplayBuffer
from .dqn import DQN


class Agent:
    """DQN agent: owns the online + target networks, replay buffer, and training.

    Online network selects actions and is updated every step.
    Target network is synced every `target_sync_every` steps and computes the
    Bellman target — decoupling the target from online updates makes DQN stable.
    """

    def __init__(self, env, vision_radius, n_actions=3, buffer_capacity=20000,
                 batch_size=128, gamma=0.99, lr=1e-3, target_sync_every=500,
                 eps_start=1.0, eps_end=0.05, eps_decay_episodes=3000,
                 max_steps=500, device="cpu"):
        self.env = env
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_sync_every = target_sync_every
        self.max_steps = max_steps
        self.device = device

        vision_size = 2 * vision_radius + 1
        self.model = DQN(vision_size, vision_radius, n_directions=4,
                         num_actions=n_actions).to(device)
        self.target = DQN(vision_size, vision_radius, n_directions=4,
                          num_actions=n_actions).to(device)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # epsilon decay
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_episodes = eps_decay_episodes
        self.eps_delta = (eps_start - eps_end) / eps_decay_episodes

        self.step_count = 0

    def epsilon(self):
        """Current exploration rate, decayed linearly over eps_decay_episodes."""
        return max(self.eps_end, self.eps_start - self.eps_delta * self.step_count)

    def train_episode(self):
        """Run one episode and return the total reward."""
        state = self.env.reset()
        done = False
        total_reward = 0.0
        steps_taken = 0

        while not done and steps_taken < self.max_steps:
            eps = self.epsilon()
            action = self.model.select_action(state, eps, self.device)

            next_state, reward, done = self.env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps_taken += 1

            # One gradient step if we have enough samples
            self._learn()

            # Periodically sync target network
            if self.target_sync_every > 0 and self.step_count % self.target_sync_every == 0:
                self.target.load_state_dict(self.model.state_dict())

            self.step_count += 1
            if self.step_count % 1000 == 0:
                print(f"  steps={self.step_count}  eps={eps:.3f}  buffer={len(self.replay_buffer)}")

        return total_reward

    def _learn(self):
        """One gradient update from a random batch of experiences."""
        if len(self.replay_buffer) < self.batch_size:
            return

        (vision, direction), (actions,), (rewards,), \
            (next_vision, next_direction), (dones,) = self.replay_buffer.sample(self.batch_size)

        vision = torch.tensor(vision, device=self.device)
        direction = torch.tensor(direction, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)
        next_vision = torch.tensor(next_vision, device=self.device)
        next_direction = torch.tensor(next_direction, device=self.device)
        dones = torch.tensor(dones, device=self.device)

        # Q(s, a) for the taken actions
        q_values = self.model(vision, direction)                        # (B, 3)
        q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)   # (B,)

        # max_{a'} Q_target(s', a') — no grad through the target network
        with torch.no_grad():
            next_q = self.target(next_vision, next_direction).max(dim=1).values

        # Bellman target; terminal states have no future reward
        target = rewards + self.gamma * next_q * (1 - dones)

        # MSE loss between prediction and target
        loss = torch.nn.functional.mse_loss(q_taken, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, num_episodes):
        """Run episodes with epsilon=0 (pure greedy). Return avg reward."""
        total = 0
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            steps_taken = 0
            while not done and steps_taken < self.max_steps:
                action = self.model.select_action(state, 0.0, self.device)
                state, reward, done = self.env.step(action)
                total += reward
                steps_taken += 1
        return total / num_episodes
