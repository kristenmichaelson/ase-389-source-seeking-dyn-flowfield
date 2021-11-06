import numpy as np
import math
# import assign_pairs
import random
import pygame

class Evader():
    def __init__(self, x, y, id, size=10):
        self.x = x
        self.y = y
        self.size = size # px
        self.colour = (0, 0, 255)
        self.thickness = 1 # px
        self.speed = 0.25
        self.ID = id
        # self.angle = 0

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
        print(self.x)
        print(self.y)
        self.x += dt * vx
        self.y += dt * vy
        print(self.x)
        print(self.y)


class Pursuer():
    def __init__(self, x, y, size=10):
        self.x = x
        self.y = y
        self.size = size
        self.colour = (255, 0, 0)
        self.thickness = 1
        self.capturing_radius = 10
        self.I_a = list(range(m))
        self.I_t = []

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
    
    # check if the pursuer can capture assigned evader
    def is_within_reach(self, ex, ey) 
        # tune the parameter r 
        return abs([self.x self.y] - [ex ey])^2 <= capturing_radius #*exp(-a*(t-t0)))^2

    # Neighbours to achieve local coordination among the pursuers
    def is_neighbor(self, px, py)
        abs([self.x self.y] - [px py])^2 <= coordination_radius #*exp(-a*(t-t0)))^2

    def is_assigned(self)
        return len(self.I_a) == 1

    def evader_assigned(self)
        return self.I_a

    def evader_assignment(self, ex, ey, eid)
        
        # capture move or velocity to make sure it stays within reach and capturability
        self.I_a = eid 
        # Do we need to store position of evader as well or just the ID 

     #input: list of pursuers up for tie break
     # pick one of the pursuers randomly
    def tb(self, list(Evader))
        

    

def task_assigment(A,I):
    #for every pursuer 
    for pi in range(len(I))
        Select evader s_i among those that satisfy 2a
        if !is_taken(s_i, I[s_i][])
            I[s_i]][1] =
            update C_i and N_i 
            I[s_i]][2] =
        else if tie break 
            tie break

#This is to generate some velocity field
def vel_form(centers, dim, list_):
    u, v = np.zeros((dim, dim)), np.zeros((dim, dim))
    x, y = np.meshgrid(np.linspace(0,dim,dim+1),np.linspace(0,dim,dim+1))
    for i in range(len(centers)):
        c1, c2 = centers[i]
        cen = (7, 7)
        u1 = -(x-c1)/((x-c1)**2 + (y-c2)**2)**(32/4)
        u1[c2, c1] = 0
        v1 = -(y-c2)/((x-c1)**2 + (y-c2)**2)**(32/4)
        v1[c2, c1] = 0
        u, v = u + list_[i]*u1[1:, 1:].copy(), v + list_[i]*v1[1:, 1:].copy()
    return u, v
    

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
    
    # list of initial positions (X = pursuers, Y = evaders)
    X = # [tuple(float, float), ...]
    Y = # [tuple(float, float), ...]

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
        e = Evader(x, y, ii)
        E.append(e)

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
        #A = [(0,0), (1,1), (2,2), (3,3)] # hard-coded assignments for now
        A = [(-1,-1) for i in range(n)]
        # Control inputs u are computed
        # Ui = [tuple(float, float)] # 2D control inputs for pursuers
        # Uk = [tuple(float, float)] # 2D control inputs for evaders
        I = np.empty((n,),dtype=object)
        for i,v in enumerate(I): 
            I[i]=[[k for k in range(m)],v]

        l=0
        for v in I: 
            v.append(l)
            l+=1
        task_assigment(I, A)
        
        # evaders that satisfy e2a
        e2a = []
        for p_ind in range(n)
            for e_ind in range(m)
                if (P[p_ind].is_within_reach(E[e_ind].x, E[e_ind].y) & (len(P[p_ind]) > 1) & (E[e_ind].ID in P[p_ind].I_a)
                    e2a.append(E[e_ind])

            if len(e2a) > 1
                P[p_ind].tiebreak(e2a) 
            else if len(e2a) == 1
                P[p_ind].evader_assignment(e2a[1].x,e2a[1].y,e2a[1].ID)

        # Integrate dynamics
        for p_ind, e_ind in A:
            e = E[e_ind]
            vx, vy = P[p_ind].vel(e.x, e.y, t, t0)
            P[p_ind].move(vx, vy, dt)
        
        print(E[0].x)

        for ii in range(m):
            vx, vy = E[ii].vel()
            E[ii].move(vx, vy, dt)

        print(E[0].x)

        # Visualize
        [p.display(screen, scale, width) for p in P]
        [e.display(screen, scale, width) for e in E]
        pygame.display.flip()
        
        # Current time
        t = t + dt

        wait = input("Press Enter to continue...")
        
        
    

if __name__ == "__main__":
	play_game()