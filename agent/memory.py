import random
from collections import deque
import numpy as np


class ReplayBuffer:
    """Store (state, action, reward, next_state, done) experience tuples.

    `state` / `next_state` are `State` dataclasses (vision + direction).
    Random sampling breaks correlation between consecutive experiences.
    """

    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Stack the raw vision arrays and directions separately.
        vision = np.stack([s.vision for s in states])
        direction = np.array([s.direction for s in states], dtype=np.int64)
        next_vision = np.stack([s.vision for s in next_states])
        next_direction = np.array([s.direction for s in next_states], dtype=np.int64)

        return (vision, direction), (actions,), (np.array(rewards, dtype=np.float32),), \
               (next_vision, next_direction), (np.array(dones, dtype=np.float32),)

    def __len__(self):
        return len(self.buffer)
