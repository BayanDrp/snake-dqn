import torch
from .network import Actor, Critic
from .memory import PPOMemory


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.0003):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr
        )

    def state_to_tensor(self, state):
        state_vector = state.vision.flatten()
        state_vector = torch.cat([
            torch.tensor(state_vector, dtype=torch.float32),
            torch.tensor([state.direction], dtype=torch.float32),
        ])
        return state_vector.unsqueeze(0)

    def select_action(self, state):
        state = self.state_to_tensor(state)
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(state)

        return action.item(), log_prob.detach(), value.detach()

    def train_episode(self, env, max_steps=500):
        memory = PPOMemory()
        state = env.reset()
        total_reward = 0

        for _ in range(max_steps):
            action, log_prob, value = self.select_action(state)
            state_vector = self.state_to_tensor(state).squeeze(0).numpy()
            next_state, reward, done = env.step(action)

            memory.store(state_vector, action, reward, done, log_prob, value)
            total_reward += reward
            state = next_state

            if done:
                break

        next_value = 0 if done else self.critic(
            self.state_to_tensor(state)
        ).squeeze(-1).detach()
        memory.calculate_returns(next_value)
        memory.calculate_advantages(next_value)
        self.update(memory)

        return total_reward

    def update(self, memory, gamma=0.99, lam=0.95, batch_size=64):
        for state, action, log_prob, returns, advantages in memory.get_batches(batch_size):
            self.update_actor(state, action, log_prob, advantages)
            self.update_critic(state, returns)

    def update_actor(self, state, action, old_log_prob, advantages):
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        new_log_prob = dist.log_prob(action)

        ratio = torch.exp(new_log_prob - old_log_prob)
        surrogate1 = ratio * advantages
        surrogate2 = (
            torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantages
        )
        actor_loss = -torch.min(surrogate1, surrogate2).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def update_critic(self, state, returns):
        value = self.critic(state).squeeze(-1)
        critic_loss = (value - returns).pow(2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()