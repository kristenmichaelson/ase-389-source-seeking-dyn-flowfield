import numpy as np
import math
import assign_pairs
import random

class Evader():
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size
        self.colour = (0, 0, 255)
        self.thickness = 1
        self.speed = 0
        self.angle = 0

    ## Helper function to visualize 
    def display(self):
        pygame.draw.circle(screen, self.colour, (int(self.x), int(self.y)), self.size, self.thickness)

    ## Evader movement defined by kinematic model
    def move(self):
        self.x += math.sin(self.angle) * self.speed
        self.y -= math.cos(self.angle) * self.speed



# Helper functions
def is_neighbor(x_i, x_j) # for pursuer-pursuer

def is_within_reach(x_i, y_k) # for pursuer-evader

def tb(list(int)) #input: list of pursuers up for tie break
    # pick one of the pursuers randomly

def evader_velocities(m):
    ## Defined an arbitrary height and width used to define the size of the 
    ##  display screen later on 

    width, height = 570,570
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
        

def play_game():
    print("Playing...")
    n = 4 # number of pursuers (0 to n-1)
    m = 4 # number of evaders (0 to m-1)
    
    # list of initial positions (X = pursuers, Y = evaders)
    X = # [tuple(float, float), ...]
    Y = # [tuple(float, float), ...]
    
    
    dt = 0.01
    
    is_game_over = False
    
    t0 = 0.0
    t = t0
    while not is_game_over:
        # Assign pursuer-evader pairs
        # Output: 
        # A = [(int: P, int: E), ...]
        # I = list of pairs [(I^a(list: int), I^t(list: int)), ...]
        # Ex:
        # I[0][0]: pursuer 0's I^a list
        # I[4][1]: pursuer 4's I^t list 
        
        # Control inputs u are computed
        # Ui = [tuple(float, float)] # 2D control inputs for pursuers
        # Uk = [tuple(float, float)] # 2D control inputs for evaders


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
            
        # Integrate dynamics (ie x(t+dt) = x(t) + dt * u(t))
        for p in range(n):
            X[p][0] += dt * Ui[p][0]
            X[p][1] += dt * Ui[p][1]
        
        # Visualize
        
        # Current time
        t = t + dt
        
        
    

if __name__ == "__main__":
	play_game()