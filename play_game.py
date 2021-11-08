import numpy as np
import math
# import assign_pairs
import random
import pygame

class Evader():
    def __init__(self, x, y,angle,speed,size=100):
        self.x = x
        self.y = y
        self.size = size # px
        self.colour = (0, 0, 255)
        self.thickness = 1 # px
        self.speed = speed
        self.angle = angle


    def m(self):
        x = int(np.cos(self.angle) * 100) + 300
        y = int(np.sin(self.angle) * 100) + 300
        return x,y

    ## Helper function to visualize 
    def display(self, screen, scale, w):
        pygame.draw.circle(screen, self.colour, (int(scale * self.x + w), int(-scale * self.y + w)), self.size, self.thickness)

    ## Compute velocity (circular motion)
    def vel(self):
        a = np.arctan2(self.x, self.y)
        vx = self.speed * np.cos(a)
        vy = self.speed * np.sin(a)
        return vx, vy

    ## Evader movement defined by kinematic model
    def move(self, vx, vy, dt):
        # print(self.x)
        # print(self.y)
        self.x += dt * vx
        self.y += dt * vy
        # print(self.x)
        # print(self.y)


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
        a = 1.0 # have to tune these (see Thm 1, Zavlanos and Pappas 2007)
        R = 1.0
        K = 1.0
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

'''
# Helper functions
def is_neighbor(x_i, x_j) # for pursuer-pursuer

def is_within_reach(x_i, y_k) # for pursuer-evader

def tb(list(int)) #input: list of pursuers up for tie break
    # pick one of the pursuers randomly
'''

'''
def evader_velocities(m):
    ## Defined an arbitrary height and width used to define the size of the 
    ##  display screen later on 

    #width, height = 570,570
    evaders_list = []
    velocities = [None for i in range(m)]

    ## Loop assigns random eveder dynamics for starting positions and speed.
    for n in range(m):
        size = 10  
        x_vel = random.randint(size, width-size)
        y_vel = random.randint(size, height-size)
        evader = Evader(x_vel, y_vel, size)
        evader.speed = random.random()
        evader.angle = random.uniform(0, math.pi*2)
        evaders_list.append(evader)

    for i in range (len(evaders_list)):
        velocities[i] = (evaders_list[i].x,evaders_list[i].y)
    return velocities
'''
        

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

    # E = [Evader(scale * 50 + 40, scale * 500 + 400,2,0.2)]
    ## Argument 
    ## x,y, angle, speed. 

    E = [Evader(500,100,2,0.05),Evader(200,200,5,0.05),Evader(300,300,9,0.05),Evader(400,400,10,0.05)]
    
    # R = 2.0
    # for ii in range(m):
    #     a = (2 * math.pi / m) * ii
    #     # x = R * np.cos(a)
    #     # y = R * np.sin(a)
    #     x = int(np.cos(a) * 100) + 100
    #     y = int(np.sin(a) * 100) + 100
    #     e = Evader(x, y)
    #     E.append(e)


    dt = 0.01
  
    
    is_game_over = False
    
    t0 = 0.0
    t = t0

    j = 0
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

        # Control inputs u are computed
        # Ui = [tuple(float, float)] # 2D control inputs for pursuers
        # Uk = [tuple(float, float)] # 2D control inputs for evaders

        '''
        E = evader_velocities(m)

        Ui = [(0,0) for pi in range(n)]
        a = 1.0 # have to tune these (see Thm 1, Zavlanos and Pappas 2007)
        R = 1.0
        K = 1.0
        for (p,e) in A:
            px = X[p][0]
            py = X[p][1]
            ex = Y[e][0]
            ey = Y[e][1]
            gamma = math.sqrt((px - ex) ** 2 + (py - ey) ** 2)
            r = R * math.exp(-a * (t - t0))
            beta = r ** 2 - gamma ** 2
            Ui[p][0] = - K * (1 / beta ** 2) * 2 * (px - ex)
            Ui[p][1] = - K * (1 / beta ** 2) * 2 * (py - ey)
        '''

        # Integrate dynamics
        for p_ind, e_ind in A:
            e = E[e_ind]
            vx, vy = P[p_ind].vel(e.x, e.y, t, t0)
            P[p_ind].move(vx, vy, dt)
        
        # print(E[0].x)

        screen.fill((255, 255, 255))

        for ii in range(len(E)):
            # E[ii].display(screen,scale,width)
            pygame.draw.circle(screen, E[ii].colour,(E[ii].x,E[ii].y), E[ii].size, E[ii].thickness)
            vx, vy = E[ii].vel()
            E[ii].x = E[ii].m()[0]
            E[ii].y = E[ii].m()[1]
            E[ii].angle += E[ii].speed
           

        # print(E[0].x)

        # Visualize
        # [p.display(screen, scale, width) for p in P]
        # [e.display(screen, scale, width) for e in E]

        # screen.fill((255, 255, 255))

        clock.tick(30)

        pygame.display.update()
      
        # Current time
        t = t + dt

        # wait = input("Press Enter to continue...")

    pygame.quit()
        
        
    

if __name__ == "__main__":
	play_game()