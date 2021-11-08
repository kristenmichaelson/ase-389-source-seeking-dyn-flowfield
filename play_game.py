import numpy as np
import math
# import assign_pairs
import random
import pygame
import time

class Evader():
    def __init__(self, x, y, size=10):
        self.x = x
        self.y = y
        self.size = size # px
        self.colour = (0, 0, 255)
        self.thickness = 1 # px
        self.speed = 2
        # self.angle = 0

    ## Helper function to visualize 
    def display(self, screen, scale, w):
        pygame.draw.circle(screen, self.colour, (int(scale * self.x + w / 2), int(-scale * self.y + w / 2)), self.size, self.thickness)

    ## Compute velocity (circular motion)
    def vel(self):
        a = np.arctan2(self.y, self.x)
        vx = -self.speed * np.sin(a)
        vy = self.speed * np.cos(a)
        return vx, vy

    ## Evader movement defined by kinematic model
    def move(self, vx, vy, dt):
        self.x += dt * vx
        self.y += dt * vy


class Pursuer():
    def __init__(self, x, y, size=10):
        self.x = x
        self.y = y
        self.size = size
        self.colour = (255, 0, 0)
        self.thickness = 1

    ## Helper function to visualize 
    def display(self, screen, scale, w):
        pygame.draw.circle(screen, self.colour, (int(scale * self.x + w / 2), int(-scale * self.y + w / 2)), self.size, self.thickness)

    ## Compute velocity (using strategy, position (ex, ey) of assigned evader)
    def vel(self, ex, ey, t, t0):
        a = 0.25 # have to tune these (see Thm 1, Zavlanos and Pappas 2007)
        R = 4.0
        K = 2.0
        gamma = math.sqrt((self.x - ex) ** 2 + (self.y - ey) ** 2)
        r = R * math.exp(-a * (t - t0))
        beta = r ** 2 - gamma ** 2
        vx = - K * (1 / beta ** 2) * 2 * (self.x - ex)
        vy = - K * (1 / beta ** 2) * 2 * (self.y - ey)
        return vx, vy

    ## Pursuer movement defined by kinematic model
    def move(self, vx, vy, dt):
        self.x += dt * vx
        self.y += dt * vy
        

def play_game():
    print("Playing...")
    n = 4 # number of pursuers (0 to n-1)
    m = 4 # number of evaders (0 to m-1)

    # Set up display
    # https://www.pygame.org/docs/tut/PygameIntro.html
    pygame.init()
    size = width, height = 600, 600 # display size 500 x 500 px
    screen = pygame.display.set_mode(size)
    scale = width / 6
    clock = pygame.time.Clock()
    
    # Wait 3 seconds (for screen recording)
    time.sleep(3)

    # list of initial positions (X = pursuers, Y = evaders)
    # X = # [tuple(float, float), ...]
    # Y = # [tuple(float, float), ...]

    # Initialize pursuers, evaders
    # Field is 6m x 6m, centered at the origin
    # Pursuers start in box [-2, -2] to [-1, -1]
    # Evaders start on a circle of radius R
    P = []
    for ii in range(n):
        x = random.random() - 2.0
        y = random.random() - 2.0
        p = Pursuer(x, y)
        P.append(p)

    E = []
    R = 2.0
    for ii in range(m):
        a = (2 * math.pi / m) * ii
        x = R * np.cos(a)
        y = R * np.sin(a)
        e = Evader(x, y)
        E.append(e)

    dt = 0.01
    
    is_game_over = False
    
    t0 = 0.0
    t = t0
    while not is_game_over:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Assign pursuer-evader pairs
        # Output: 
        # A = [(int: P, int: E), ...]
        # I = list of pairs [(I^a(list: int), I^t(list: int)), ...]
        # Ex:
        # I[0][0]: pursuer 0's I^a list
        # I[4][1]: pursuer 4's I^t list 
        A = [(0,0), (1,1), (2,2), (3,3)] # hard-coded assignments for now

        # Integrate dynamics
        for p_ind, e_ind in A:
            e = E[e_ind]
            vx, vy = P[p_ind].vel(e.x, e.y, t, t0)
            P[p_ind].move(vx, vy, dt)

        for ii in range(m):
            vx, vy = E[ii].vel()
            E[ii].move(vx, vy, dt)

        # Visualize
        screen.fill((255,255,255))
        [p.display(screen, scale, width) for p in P]
        [e.display(screen, scale, width) for e in E]
        clock.tick(30)
        pygame.display.update()
        
        # Current time
        t = t + dt

        
        
    

if __name__ == "__main__":
	play_game()