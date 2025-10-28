import pygame

class Block:
    def __init__(self, screen, x, y, passable = False, flag = 'w'):
        self.screen = screen
        self.x = x
        self.y = y
        self.passable = passable
        self.flag = flag
        self.surface = pygame.Surface((80, 80))
        self.surface.fill("white" if self.passable else "black")
        self.distance = -1  # for pathfinding use
    
    def draw(self, box_size):
        #self.surface.blit(self.surface, (self.x*box_size, self.y*box_size))
        pygame.draw.rect(self.screen, "white" if self.passable else "black",
                        [self.x * box_size, self.y * box_size, box_size, box_size])
        
    def changeStatus(self, flag=None):
        if not flag:
            self.passable = not self.passable
            self.flag = 'p' if self.passable else 'w'
        else:
            self.passable = True
            self.flag = flag
        self.surface.fill("white" if self.passable else "black")
        return self

class GoalPoint(Block):
    def __init__(self, screen, x, y, passable=True, flag='g'):
        super().__init__(screen, x, y, passable, flag)
        self.surface.fill("green")

    def draw(self, box_size):
        pygame.draw.rect(self.screen, "green",
                         [self.x * box_size, self.y * box_size, box_size, box_size])

class StartPoint(Block):
    def __init__(self, screen, x, y, passable=True, flag='s'):
        super().__init__(screen, x, y, passable, flag)
        self.surface.fill("blue")

    def draw(self, box_size):
        pygame.draw.rect(self.screen, "blue",
                         [self.x * box_size, self.y * box_size, box_size, box_size])