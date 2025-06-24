import pygame
from env import Env
from cell import Cell

pygame.init()

env = Env()

screen = pygame.display.set_mode(env.SCREEN_SIZE, pygame.RESIZABLE)
clock = pygame.time.Clock()


running = True
while running:
    screen.fill("#000000")

    if pygame.mouse.get_pressed()[0]:
        pass

    if pygame.key.get_pressed()[pygame.K_1]:
        pass

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_2:
                pass

        if event.type == pygame.MOUSEBUTTONDOWN:
            if pygame.mouse.get_pressed()[0]:
                pass

    pygame.display.flip()
    clock.tick(env.TARGET_FPS)


pygame.quit()