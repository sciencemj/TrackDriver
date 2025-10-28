import pygame
import math

class Car:
    def __init__(self, screen, x, y, speed = 0, angle = 0, max_speed = 10, rotate_speed = 1, acceleration = 0.1, width = 30, height = 17):
        self.screen = screen
        # position & motion
        self.x = x
        self.y = y
        self.speed = speed
        self.angle = angle  # degrees
        self.max_speed = max_speed
        self.rotate_speed = rotate_speed
        self.acceleration = acceleration

        self.width = width
        self.height = height
        self.surface_original = pygame.image.load("car_image.png")
        self.surface = pygame.image.load("car_image.png")
        

    def draw(self):
        '''
        hw = self.width / 2
        hh = self.height / 2
        rad = math.radians(self.angle)
        c = math.cos(rad)
        s = math.sin(rad)

        def rot(dx, dy):
            rx = dx * c - dy * s
            ry = dx * s + dy * c
            return (self.x + rx, self.y + ry)

        p1 = rot(-hw, -hh)
        p2 = rot(hw, -hh)
        p3 = rot(hw, hh)
        p4 = rot(-hw, hh)

        pygame.draw.polygon(self.screen, "gray", [p1, p2, p3, p4])
        pygame.draw.polygon(self.screen, "red", [p2, p3, (self.x, self.y)])
        '''
        

    
    def move(self):
        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.speed
        self.y += math.sin(rad) * self.speed
        self.surface = pygame.transform.rotate(self.surface_original, -self.angle)

    def raycast(self, track, box_size, max_distance=1000, step=4):
        """
        Cast 5 rays from the car center at angles [-60, -30, 0, 30, 60] degrees relative
        to the car's current angle. Step along each ray in `step` pixel increments until
        a non-passable block or the map boundary is hit or max_distance is reached.

        Args:
            track (list[list[Block]]): 2D grid of Block objects (indexed as track[row][col]).
            box_size (int): size in pixels of one grid cell.
            max_distance (int): maximum ray length in pixels.
            step (int): sampling step in pixels along the ray.

        Returns:
            list of dict: for each ray returns a dict with keys:
                - 'angle': absolute angle in degrees (world coords)
                - 'distance': distance in pixels to the hit (or max_distance)
                - 'hit': bool whether a blocking cell was hit
                - 'point': (x, y) world coordinates of the hit point (or end point)
        """
        results = []
        # relative angles to cast (degrees)
        rel_angles = [-60, -30, 0, 30, 60]
        rows = len(track)
        cols = len(track[0]) if rows > 0 else 0

        for rel in rel_angles:
            ang = self.angle + rel
            rad = math.radians(ang)
            hit = False
            hit_point = (self.x + math.cos(rad) * max_distance, self.y + math.sin(rad) * max_distance)
            distance = max_distance

            t = 0
            while t <= max_distance:
                px = self.x + math.cos(rad) * t
                py = self.y + math.sin(rad) * t

                gx = int(px // box_size)
                gy = int(py // box_size)

                # out of bounds -> treat as hit
                if gx < 0 or gy < 0 or gy >= rows or gx >= cols:
                    hit = False
                    hit_point = (px, py)
                    distance = t
                    break

                cell = track[gy][gx]
                if not cell.passable:
                    hit = True
                    hit_point = (px, py)
                    distance = t
                    break

                t += step

            results.append({
                'angle': ang,
                'distance': distance,
                'hit': hit,
                'point': hit_point,
            })

        # store last results on the car for debugging/visualization
        self.raycast_results = results
        return results