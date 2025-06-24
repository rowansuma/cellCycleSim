import numpy as np
import pygame

class Cell:
    def __init__(self, env, pos, vel, mass):
        self.env = env
        self.pos = pos
        if not isinstance(pos, np.ndarray):
            self.pos = np.array(pos)
        self.vel = vel
        if not isinstance(vel, np.ndarray):
            self.vel = np.array(vel)
        self.mass = mass

        self.color = self.env.CELL_COLOR
        self.radius = self.env.CELL_RADIUS

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, self.pos, self.radius)

    def update(self):
        pass