#! /usr/bin/env python3
import pygame
import sys
from objects.blocks import *
        

if __name__ == "__main__":
    size = input("enter the size of map!(W H): ")
    width, height = map(int, size.split(' '))
    output_file = input("output file name: ")
    # pygame init
    box_size = 80
    pygame.init()
    screen = pygame.display.set_mode((width * box_size, height * box_size))
    clock = pygame.time.Clock()
    running = True

    blocks = [[Block(screen, x, y) for x in range(width)] for y in range(height)]
    # game loop
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                x, y = map(lambda i: i//box_size, pos)
                if event.button == 1:
                    blocks[y][x].changeStatus()
                elif event.button == 2:
                    blocks[y][x] = StartPoint(screen, x, y)
                elif event.button == 3:
                    blocks[y][x] = GoalPoint(screen, x, y)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    with open('tracks/' + output_file + '.track', 'w') as f:
                        f.write(f'{width} {height}\n')
                        for y in blocks:
                            f.write(''.join(map(lambda b: b.flag, y))+'\n')
                    quit()


        # fill the screen with a color to wipe away anything from last frame
        screen.fill("lime")

        # RENDER YOUR GAME HERE
        for lines in blocks:
            for block in lines:
                block.draw(box_size)
        # flip() the display to put your work on screen
        pygame.display.flip()

        clock.tick(60)  # limits FPS to 60

    pygame.quit()