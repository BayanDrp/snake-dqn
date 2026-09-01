import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """MLP that maps (vision window + direction) -> Q-values for [left, straight, right].

    The vision window is flattened and concatenated with the (one-hot) direction,
    then fed through a small MLP. A small CNN is unnecessary here: the 7x7 local
    window is tiny and we lose nothing by flattening it.
    """

    def __init__(self, vision_size, vision_radius, n_directions=4, num_actions=3,
                 hidden=(64, 64)):
        super().__init__()
        self.vision_size = vision_size          # e.g. 7 (window is 7x7)
        self.in_dim = vision_size * vision_size + n_directions

        self.fc1 = nn.Linear(self.in_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], num_actions)

    def forward(self, vision, direction):
        """vision: (batch, V, V) int/normalized, direction: (batch,) int."""
        v = vision.float().view(vision.size(0), -1)          # (batch, V*V)
        v = v / 3.0                                          # normalize 0..3 -> 0..1
        d = F.one_hot(direction, num_classes=4).float()      # (batch, 4)

        x = torch.cat([v, d], dim=-1)                        # (batch, V*V + 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)                                   # (batch, 3)

    def select_action(self, state, epsilon, device="cpu", n_actions=3):
        """epsilon-greedy action from a single State (no batch)."""
        if np.random.random() < epsilon:
            return np.random.randint(n_actions)

        vision = torch.tensor(state.vision, dtype=torch.float32,
                              device=device).unsqueeze(0)
        direction = torch.tensor([state.direction], dtype=torch.long,
                                 device=device)
        with torch.no_grad():
            q = self.forward(vision, direction)               # (1, 3)
        return int(q.argmax(dim=-1).item())
