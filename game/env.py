import numpy as np
from dataclasses import dataclass


@dataclass
class State:
    """State fed to the DQN: local vision window around head + direction."""
    vision: np.ndarray  # (2R+1, 2R+1) int8: 0 empty, 1 wall, 2 body, 3 food
    direction: int      # 0 up, 1 right, 2 down, 3 left


# Absolute directions: 0 up, 1 right, 2 down, 3 left
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# Relative actions: 0=left, 1=straight, 2=right
TURN = {0: -1, 1: 0, 2: 1}


class Snake:
    """Owns its own body, direction, and movement. Head is the LAST body cell."""

    def __init__(self, start=(4, 4), direction=1, length=3):
        # Build a horizontal body: head at `start`, tail to its left.
        self.direction = direction
        # cells ordered tail -> head
        self.body = [
            (start[0], start[1] - (length - 1 - i)) for i in range(length)
        ]

    @property
    def head(self):
        """(row, col) of the current head."""
        return self.body[-1]

    def relative_move(self, action):
        """Relative action (0 l, 1 straight, 2 r) -> new absolute direction.

        Returns the NEW head position WITHOUT mutating the body, so the env
        can check collisions before committing the move.
        """
        self.direction = (self.direction + TURN[action]) % 4
        dr, dc = DIRS[self.direction]
        hr, hc = self.head
        return (hr + dr, hc + dc)

    def advance(self, new_head, grow=False):
        """Commit a move: push new head. If not growing, drop the tail."""
        self.body.append(new_head)
        if not grow:
            self.body.pop(0)


class SnakeEnv:
    """The game world. Owns the grid + food + scoring, orchestrates a Snake."""

    def __init__(self, grid_size=12, vision_radius=3,
                 reward_food=10.0, reward_death=-10.0, reward_step=-0.01):
        self.grid_size = grid_size
        self.vision_radius = vision_radius
        self.reward_food = reward_food
        self.reward_death = reward_death
        self.reward_step = reward_step
        self.reset()

    def reset(self) -> State:
        """Rebuild board, snake, food. Return initial state."""
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        mid = self.grid_size // 2
        self.snake = Snake(start=(mid, mid), direction=1, length=3)
        for r, c in self.snake.body:
            self.grid[r, c] = 2  # body

        self.food = self._spawn_food()
        self.score = 0
        self.steps = 0
        self.done = False
        return self._get_state()

    def _spawn_food(self):
        """Place food at a random empty cell. Returns None if board full."""
        empty = np.argwhere(self.grid == 0)
        if len(empty) == 0:
            return None
        r, c = empty[np.random.randint(len(empty))]
        self.grid[r, c] = 3  # food
        return (int(r), int(c))

    def _get_state(self) -> State:
        """Local vision window around the head + current direction."""
        r, c = self.snake.head
        R = self.vision_radius
        size = 2 * R + 1
        vision = np.ones((size, size), dtype=np.int8)  # default = wall

        for dr in range(-R, R + 1):
            for dc in range(-R, R + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    vision[dr + R, dc + R] = self.grid[nr, nc]

        return State(vision=vision, direction=self.snake.direction)

    def step(self, action: int):
        """Apply relative action. Return (next_state, reward, done)."""
        if self.done:
            return self._get_state(), self.reward_death, True

        # Compute candidate head (no mutation yet)
        new_head = self.snake.relative_move(action)

        nr, nc = new_head
        hit_wall = not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size)
        hit_self = hit_wall or self.grid[nr, nc] == 2

        if hit_self:
            self.done = True
            return self._get_state(), self.reward_death, True

        # Eat?
        grow = new_head == self.food

        # Commit the move
        self.snake.advance(new_head, grow=grow)
        self.grid[nr, nc] = 2

        if grow:
            self.score += 1
            self.food = self._spawn_food()
            reward = self.reward_food
        else:
            # Shrank: clear the vacated tail cell
            tr, tc = self.snake.body[0]
            self.grid[tr, tc] = 0
            reward = self.reward_step

        self.steps += 1
        return self._get_state(), reward, self.done

    