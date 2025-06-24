import pygame

pygame.init()

screenSize = (800, 600)
fps = 60

screen = pygame.display.set_mode(screenSize, pygame.RESIZABLE)
clock = pygame.time.Clock()


running = True
while running:
    screen.fill("#ffffff")

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
    clock.tick(fps)


pygame.quit()