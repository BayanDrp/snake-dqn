import pygame
import numpy as np
from game.env import SnakeEnv, State

# Constants
width = 600
height = 600

# colors 
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)


class SnakeGame:
    def __init__(self, grid_size=12, vision_radius=3):
        self.env = SnakeEnv(grid_size=grid_size, vision_radius=vision_radius)
        self.cell_size = width // grid_size
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.done = False

    def render_food_collected(self):
        font = pygame.font.Font(None, 36)
        text = font.render(f"Food Collected: {self.env.score}", True, white)
        self.screen.blit(text, (10, 10))
        
    def draw_grid(self):
        for x in range(0, width, self.cell_size):
            for y in range(0, height, self.cell_size):
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, white, rect, 1)

    def draw_snake(self):
        for segment in self.env.snake.body:
            rect = pygame.Rect(segment[1] * self.cell_size,
                               segment[0] * self.cell_size,
                               self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, green, rect)

    def draw_food(self):
        food_pos = self.env.food
        rect = pygame.Rect(food_pos[1] * self.cell_size,
                           food_pos[0] * self.cell_size,
                           self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, red, rect)

    def render(self):
        self.screen.fill(black)
        self.draw_grid()
        self.draw_snake()
        self.draw_food()
        self.render_food_collected()
        pygame.display.flip()

    def update(self, action):
        next_state, reward, done = self.env.step(action)
        self.render()
        self.clock.tick(10)  # Control the speed of the game
        return next_state, reward, done