from collections import deque
from gymnasium import spaces
import gymnasium
import pygame
import numpy as np
from objects.blocks import *
from objects.car import Car

class RaceEnvironment(gymnasium.Env):

    def readTrack(self, track_lines, screen, box_size):
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
    def __init__(self, track_file, box_size=80, start_angle=90):
        super(RaceEnvironment, self).__init__()
        self.box_size = box_size
        self.start_angle = start_angle

        # Load track
        with open(track_file, 'r') as f:
            track_lines = f.readlines()
        
        pygame.init()
        self.screen = pygame.display.set_mode((1280, 720))
        self.track, self.start_point, self.goal_point = self.readTrack(track_lines[1:], self.screen, box_size)

        # Define action and observation space
        # Actions: [steering (-1 to 1), throttle (0 to 1)]
        self.action_space = spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float64)

        # Observations: 5 ray distances + speed + angle
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(7,), dtype=np.float64)

        # Initialize car
        self.car = Car(self.screen,
                       x=self.start_point[0],
                       y=self.start_point[1],angle=self.start_angle)

    def reset(self, seed=None, options=None):
        #pygame.init()
        #self.screen = pygame.display.set_mode((len(self.track[0]) * self.box_size,
        #                                       len(self.track) * self.box_size))
        # Reset car position and state
        self.car.x = self.start_point[0]
        self.car.y = self.start_point[1]
        self.car.speed = 0
        self.car.angle = self.start_angle
        return self._get_observation(), {} # observation and info
    
    def step(self, action):
        if isinstance(action[0], np.float64): steering, throttle = action
        else: steering, throttle = action[0]
        # Update car state based on action
        self.car.angle += steering * self.car.rotate_speed
        self.car.speed += throttle * self.car.acceleration
        self.car.speed = max(0, min(self.car.speed, self.car.max_speed))
        self.car.move()

        observation = self._get_observation()
        reward = self._compute_reward()
        done = self._check_done()
        info = {}

        return observation, reward, done, False, info
    
    def _get_observation(self):
        ray_distances = list(map(lambda x: x['distance'], self.car.raycast(self.track, self.box_size)))
        return np.array(ray_distances + [self.car.speed, self.car.angle], dtype=np.float32)
    
    def _compute_reward(self):
        # +0.1 if speed > 0 and survived, -1 if hit wall, +100 if reached goal
        reward = 0.0
        if self.car.speed > 0:
            reward += 0.1
        car_block_x = int(self.car.x // self.box_size)
        car_block_y = int(self.car.y // self.box_size)
        if car_block_x == int(self.goal_point[0] // self.box_size) and \
           car_block_y == int(self.goal_point[1] // self.box_size):
            reward += 100.0
        
        return reward
    
    def _check_done(self):
        car_block_x = int(self.car.x // self.box_size)
        car_block_y = int(self.car.y // self.box_size)
        # Check for collision with non-passable block
        car_rect = self.car.surface.get_rect(center=(self.car.x, self.car.y))
        for lines in self.track:
            for block in lines:
                if not block.passable:
                    block_rect = block.surface.get_rect(topleft=(block.x*self.box_size, block.y*self.box_size))
                    if car_rect.colliderect(block_rect):
                        return True
        # Check for reaching goal
                if block.flag == 'g':
                    block_rect = block.surface.get_rect(topleft=(block.x*self.box_size, block.y*self.box_size))
                    if car_rect.colliderect(block_rect):
                        return True
        return False
    
    def render(self, mode='human'):
        # Render the environment to the screen
        self.screen.fill("gray")
        for lines in self.track:
            for block in lines:
                self.screen.blit(block.surface, (block.x*self.box_size, block.y*self.box_size))
        car_rect = self.car.surface.get_rect(center=(self.car.x, self.car.y))
        self.screen.blit(self.car.surface, car_rect)
        pygame.display.flip()
    