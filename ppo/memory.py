import torch
import numpy as np



class PPOMemory:
    """Stores one on-policy rollout for PPO."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.returns = []
        self.advantages = []

    def store(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def calculate_returns(self, next_value, gamma=0.99):
        self.returns = []
        if torch.is_tensor(next_value):
            next_value = next_value.item()

        for i in reversed(range(len(self.rewards))):
            next_value = (
                self.rewards[i]
                + gamma * next_value * (1 - self.dones[i])
            )
            self.returns.insert(0, next_value)

    def calculate_advantages(self, next_value, gamma=0.99, lam=0.95):
        self.advantages = []
        gae = 0
        if torch.is_tensor(next_value):
            next_value = next_value.item()

        for i in reversed(range(len(self.rewards))):
            next_val = next_value if i == len(self.rewards) - 1 else self.values[i + 1]
            value = self.values[i]
            if torch.is_tensor(next_val):
                next_val = next_val.item()
            if torch.is_tensor(value):
                value = value.item()

            delta = (
                self.rewards[i]
                + gamma * next_val * (1 - self.dones[i])
                - value
            )
            gae = delta + gamma * lam * (1 - self.dones[i]) * gae
            self.advantages.insert(0, gae)

        self.advantages = torch.tensor(self.advantages, dtype=torch.float32)
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-8
        )

        self.returns = torch.tensor(self.returns, dtype=torch.float32)

    def get_batches(self, batch_size):
        n_samples = len(self.states)
        indices = torch.randperm(n_samples)
        log_probs = [
            log_prob.item() if torch.is_tensor(log_prob) else log_prob
            for log_prob in self.log_probs
        ]

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            states = torch.tensor(
                np.asarray(self.states), dtype=torch.float32
            )

            yield (
                states[batch_indices],
                torch.tensor(self.actions)[batch_indices],
                torch.tensor(log_probs, dtype=torch.float32)[batch_indices],
                self.returns[batch_indices],
                self.advantages[batch_indices],
            )

    

    