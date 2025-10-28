#! /usr/bin/env python3
import pygame
import sys
import math
import torch
from objects.blocks import *
from objects.car import Car
from collections import deque

class RegressionModel(torch.nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.f1 = torch.nn.Linear(7, 50)
        self.f2 = torch.nn.Linear(50, 30)
        self.f3 = torch.nn.Linear(30, 2)
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x):
        x = torch.relu(self.f1(x))
        x = self.dropout(x)
        x = torch.relu(self.f2(x))
        x = self.dropout(x)
        x = self.f3(x)
        # Interpret outputs as [steering, throttle]:
        # - steering in [-1, 1] (use tanh)
        # - throttle in [0, 1] (use sigmoid)
        steering = torch.tanh(x[:, 0])
        throttle = torch.sigmoid(x[:, 1])
        out = torch.stack((steering, throttle), dim=1)

        return out

def readTrack(track_lines, screen, box_size):
    """
    Parse track_lines into Block objects and compute distance (in grid steps)
    from each passable block to the nearest goal. Non-passable blocks get -1.
    Distance is 0 for the goal itself, 1 for orthogonal neighbors, etc.
    The computed distance is attached to each Block as `.distance`.
    Returns (track_list, start_point, goal_point) to match existing call-site.
    """

    # Normalize lines (remove CR/LF only, keep spaces if any)
    grid = [list(line.rstrip('\r\n')) for line in track_lines]
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0

    # Create block objects
    track_list = [[Block(screen, x, y) for x in range(w)] for y in range(h)]

    start_point = None
    goal_point = None

    # distance matrix initialized to -1 (unreachable / non-passable)
    distances = [[-1 for _ in range(w)] for _ in range(h)]
    q = deque()

    # First pass: instantiate special blocks and enqueue goals
    for i in range(h):
        for j in range(w):
            c = grid[i][j] if j < len(grid[i]) else ' '  # guard if ragged input
            if c == 'g':
                track_list[i][j] = GoalPoint(screen, j, i)
                goal_point = (j * box_size + 0.5 * box_size, i * box_size + 0.5 * box_size)
                distances[i][j] = 0
                q.append((j, i))
            elif c == 's':
                track_list[i][j] = StartPoint(screen, j, i)
                start_point = (j * box_size + 0.5 * box_size, i * box_size + 0.5 * box_size)
                # treat start as passable; distance left as -1 until found by BFS
            elif c == 'p':
                # make this block passable
                track_list[i][j].changeStatus()
            # otherwise leave default Block (assumed non-passable)

    # BFS from all goal cells to compute shortest path (4-neighbor) distances
    while q:
        x, y = q.popleft()
        current_dist = distances[y][x]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                neighbor = track_list[ny][nx]
                # skip non-passable blocks
                if not getattr(neighbor, "passable", False):
                    distances[ny][nx] = -1
                    continue
                # if not visited yet, set distance and enqueue
                if distances[ny][nx] == -1:
                    distances[ny][nx] = current_dist + 1
                    q.append((nx, ny))

    # Attach computed distances to each Block object for easy lookup later
    for i in range(h):
        for j in range(w):
            track_list[i][j].distance = distances[i][j]

    return track_list, start_point, goal_point

class CustomCriterion(torch.nn.Module):
    """
    Custom criterion that converts your previous scoring logic into a torch-compatible loss.
    It returns a scalar tensor (loss = -reward) so you can call .backward() on it.
    """
    def __init__(self):
        super(CustomCriterion, self).__init__()

    def forward(self, time, track, death_block, output=None):
        # compute reward (same logic as before)
        reward = - time / 10.0
        if death_block.flag == 'g':
            reward += 1000.0
        else:
            goal_x, goal_y = None, None
            for row in track:
                for block in row:
                    if block.flag == 'g':
                        goal_x, goal_y = block.x, block.y
                        break
                if goal_x is not None:
                    break
            if goal_x is None:
                goal_x, goal_y = death_block.x, death_block.y
            #reward -= death_block.distance * 1.
            straight_distance = math.hypot(goal_x - death_block.x, goal_y - death_block.y)
            reward -= straight_distance * 5.0
        # Convert reward to a torch loss (we minimize loss, so loss = -reward).
        # Use a tensor so .backward() works; requires_grad=True is not strictly necessary
        # here because the loss should be a function of model parameters for meaningful gradients.
        loss = torch.tensor(-reward, dtype=torch.float32, requires_grad=True)
        return loss


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("you need track file!")
        quit()

    # pygame setup
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    clock = pygame.time.Clock()
    running = True
    box_size = 80
    trackfile = sys.argv[1]
    track = []
    with open(trackfile, 'r') as f:
        x, y = map(int, f.readline().split())
        track, start_point, goal_point = readTrack(f.readlines(), screen, box_size)
    
    car = Car(screen, start_point[0], start_point[1], speed=0, angle=0)
    front = 0
    side = 0
    start_time = 0
    def init():
        start_time = pygame.time.get_ticks()
        car = Car(screen, start_point[0], start_point[1], speed=0, angle=0)
        return start_time, car
    
    model = RegressionModel()
    scoring = CustomCriterion()
    optimizer = torch.optim.Adam(model.parameters(), lr=1)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            pressed = pygame.key.get_pressed()
            
            #front = 1 if pressed[pygame.K_w] else -1 if pressed[pygame.K_s] else 0
            #side = 1 if pressed[pygame.K_d] else -1 if pressed[pygame.K_a] else 0
        
        # use model to predict front and side
        # get raycast distances
        rays = car.raycast(track, box_size, max_distance=800, step=4)
        ray_distances = []
        for r in rays:
            ray_distances.append(r['distance'] / 800)  # normalize to [0, 1]
        # prepare input tensor
        input_tensor = torch.tensor([[car.speed / car.max_speed] + [car.angle / 360] + ray_distances], dtype=torch.float32)
        output = model(input_tensor)
        side = output[0][0].item()  # steering
        front = output[0][1].item()  # throttle
        # fill the screen with a color to wipe away anything from last frame
        screen.fill("gray")

        # RENDER YOUR GAME HERE
        for lines in track:
            for block in lines:
                screen.blit(block.surface, (block.x*box_size, block.y*box_size))
        
        car.speed += car.acceleration * front
        car.angle += car.rotate_speed * side
        car.move()
        car_rect = car.surface.get_rect(center=(car.x, car.y))
        #check collision with blocks
        for lines in track:
            for block in lines:
                if not block.passable:
                    block_rect = block.surface.get_rect(topleft=(block.x*box_size, block.y*box_size))
                    if car_rect.colliderect(block_rect):
                        #simple collision response: stop the car
                        car.speed = 0
                        start_time, car = init()
                        # train model
                        # get the block where the car's center is located
                        bx = int(car.x // box_size)
                        by = int(car.y // box_size)
                        # clamp to valid indices in case of edge/rounding issues
                        if len(track) == 0:
                            death_block = None
                        else:
                            by = max(0, min(by, len(track) - 1))
                            bx = max(0, min(bx, len(track[by]) - 1))
                            death_block = track[by][bx]

                        loss = scoring((pygame.time.get_ticks() - start_time)/1000, track, death_block)
                        print("Loss on collision:", loss.item())
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                if block.flag == 'g':
                    block_rect = block.surface.get_rect(topleft=(block.x*box_size, block.y*box_size))
                    if car_rect.colliderect(block_rect):
                        print(scoring((pygame.time.get_ticks() - start_time)/1000, track, block))
                        start_time, car = init()
        # visiualize rays
        for r in rays:
            pygame.draw.line(screen, (255,0,0), (car.x, car.y), r['point'], 1)
            # optional: draw a small circle at the hit-point
            pygame.draw.circle(screen, (255,255,0), (int(r['point'][0]), int(r['point'][1])), 3)

        screen.blit(car.surface, car_rect)

        # flip() the display to put your work on screen
        pygame.display.flip()
        clock.tick(6000)  # limits FPS to 60

    pygame.quit()