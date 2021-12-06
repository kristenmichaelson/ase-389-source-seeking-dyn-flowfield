import numpy as np
import math
import random
from numpy.lib.function_base import disp
import matplotlib.pyplot as plt
import pygame
import time
from scipy.optimize import linear_sum_assignment
from math import hypot, atan2, sin, cos, pi;

# global variables
n = 7 # number of pursuers (0 to n-1)
m = 7 # number of evaders (0 to m-1)

#flow field 
dim = 30
round_ = 1
cen_list = [(7 + round_ - 1, 7+ round_ - 1), (42 - round_ + 1, 45 - round_ + 1), (6 + round_ - 1, 45 - round_ + 1)]    
ratio_list = [0.95/5.0, 0.15/5.0, 1.35/5.0]

F = 2.0 # velocity of vehicle relative to flow field

class Evader():
    def __init__(self, x, y, id, size=10):
        self.x = x
        self.y = y
        self.size = size # px
        self.colour = (0, 0, 255)
        self.thickness = 1 # px
        self.speed = 2
        self.ID = id
        self.font = pygame.font.SysFont(None, 15)
        self.text = self.font.render(str(self.ID), True, (0, 0, 0))


    ## Helper function to visualize 
    def display(self, screen, scale, w):
        pygame.draw.circle(screen, self.colour, (int(scale * self.x + w / 2), int(-scale * self.y + w / 2)), self.size, self.thickness)
        screen.blit(self.text, self.text.get_rect(center = (int(scale * self.x + w / 2), int(-scale * self.y + w / 2)) ))

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
    def __init__(self, x, y,id, size=10):
        self.x = x
        self.y = y
        self.size = size
        self.colour = (255, 0, 0)
        self.thickness = 1
        self.capturing_radius = 10
        self.coordination_radius = 10
        self.I_a = list(range(m))
        self.I_t = []
        self.C_i = []
        self.neighbours_list = []
        self.ID = id
        self.font = pygame.font.SysFont(None, 15)
        self.text = self.font.render(str(self.ID), True, (205,51,51))
        self.vx = 0
        self.vy = 0




    ## Helper function to visualize 
    def display(self, screen, scale, w):
        pygame.draw.circle(screen, self.colour, (int(scale * self.x + w / 2), int(-scale * self.y + w / 2)), self.size, self.thickness)
        screen.blit(self.text, self.text.get_rect(center = (int(scale * self.x + w / 2), int(-scale * self.y + w / 2)) ))
       

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
        
    # check if the pursuer can capture assigned evader
    def is_within_reach(self, ex, ey): 
        # tune the parameter r 
        return (self.x - ex)**2 + (self.y - ey)**2 <= self.capturing_radius^2 #*exp(-a*(t-t0)))^2
    

    # Neighbours to achieve local coordination among the pursuers
    def is_neighbor(self, px, py):
        return (self.x - px)**2 + (self.y - py)**2 <= self.coordination_radius #*exp(-a*(t-t0)))^2

    def is_assigned(self):
        return len(self.I_a) == 1

    def evader_assigned(self):
        return self.I_a

    def neighbours(self, list_of_pursuers):
        self.neighbours_list = []
        for pi in list_of_pursuers:
            if self.is_neighbor(pi.x,pi.y):
                self.neighbours_list.append(pi)
        return self.neighbours_list




WIDTH= 600 
HEIGHT = 600

x_units, y_units = 8, 8
hwidth, hheight = WIDTH / 2, HEIGHT / 2
x_scale, y_scale = WIDTH / x_units, HEIGHT / y_units

### Converts to rectangular form
def from_polar(r, theta):
    return r * cos(theta), r * sin(theta)

#### Scale the flow field
def translate_and_scale(x, y):
    return (x * x_scale) + hwidth, hheight - (y * y_scale)

def draw_arrow(A, B, surf, width=2):
    dy, dx = A[1] - B[1], A[0] - B[0]
    angle = atan2(dy, dx)

    color = (0, 0, 0)

    dist = hypot(dx, dy) / 5
    ### use to find arrow head angle #######
    x1, y1 = from_polar(dist, angle + (pi / 4))
    x2, y2 = from_polar(dist, angle - (pi / 4))
    pygame.draw.line(surf, color, A, B, width)
    pygame.draw.line(surf, color, B, (B[0] + x1, B[1] + y1), width)
    pygame.draw.line(surf, color, B, (B[0] + x2, B[1] + y2), width)


class VectorField():
    def __init__(self,function,scale,dim,step):
        self.function = function
        self.dim = dim
        self.step = step
        # self.scale = scale
        self._generate_vectors()

    def circular_velocity_field(self,scale):
        x_field, y_field = np.meshgrid(np.linspace(-dim,dim,self.step),np.linspace(-dim,dim,self.step))
        u_field = -scale*y_field/np.sqrt(x_field**2 + y_field**2)
        v_field = scale*x_field/np.sqrt(x_field**2 + y_field**2)
        return x_field, y_field, u_field, v_field

    def _generate_vectors(self):

        self.vectors = []

        x_field,y_field = np.meshgrid(np.linspace(-dim,dim,self.step),np.linspace(-dim,dim,self.step))

        # u = - self.scale * y_field/np.sqrt(x_field**2 + y_field**2)
        # v = self.scale * x_field/np.sqrt(x_field**2 + y_field**2)

        B = x_field.tolist()
        C = y_field.tolist()

        H = list(zip(B,C))
        for i in range(len(H)):
            for j in range(len(H[i][0])):
                tup = (H[i][0][j], H[i][1][j])
                dx, dy = self.function(tup[0], tup[1])
                self.vectors.append((tup[0], tup[1], tup[0] + dx / 8, tup[1] + dy / 8))


    def draw(self, surf):
        for vector in self.vectors:
            draw_arrow(translate_and_scale(vector[0], vector[1]), translate_and_scale(vector[2], vector[3]), surf)


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

def circular_velocity_field(scale):
    
    x_field, y_field = np.meshgrid(np.linspace(-dim,dim,2*dim+1),np.linspace(-dim,dim,2*dim+1))

    u_field = -scale*y_field/np.sqrt(x_field**2 + y_field**2)
    v_field = scale*x_field/np.sqrt(x_field**2 + y_field**2)
    return x_field, y_field, u_field, v_field

def min_dist(evader, list_of_pursuers): 
    min_dist = 100000
    s_i = list_of_pursuers[0]
    for pursuer in list_of_pursuers:
        dist = (pursuer.x - evader.x)**2 + (pursuer.y- evader.y)**2
        if (dist < min_dist):
            min_dist = dist
            s_i = pursuer
    return s_i

def task_assignment(P,E):

    e2a = [] # evaders that satisfy assumption 2a
    p2a = []
    for p_ind in range(n):
        for e_ind in range(m):
            if (P[p_ind].is_within_reach(E[e_ind].x, E[e_ind].y)) & (len(P[p_ind].I_a) > 1) & (E[e_ind].ID in P[p_ind].I_a):
                e2a.append(E[e_ind])
                
        if (len(e2a) >= 1):
            # select an evader from all evaders that satisfy assumption 2a, this also incorporates tie breaking 
            random_index = random.randint(0,len(e2a)-1)
            selected_evader = e2a[random_index]
            
            # identify candidate purusers 
            for p_ind in range(n):
                if (P[p_ind].is_within_reach(selected_evader.x, selected_evader.y)) & (len(P[p_ind].I_a) > 1) & (selected_evader.ID in P[p_ind].I_a):
                    p2a.append(P[p_ind])

            # select the closest pursuer 
            selected_pursuer = min_dist(selected_evader, p2a)

            #if selected evader is available, update Ia and It of all neighboring purusers
            if selected_evader.ID in selected_pursuer.I_a: 
                selected_pursuer.I_a = []
                selected_pursuer.I_a.append(selected_evader.ID)
                selected_pursuer.I_t.append(selected_evader.ID) 
                for neighbour in selected_pursuer.neighbours(P):
                    if len(neighbour.I_a) > 1 :
                        neighbour.I_a.remove(selected_evader.ID)
                    if selected_evader.ID not in neighbour.I_t:
                        neighbour.I_t.append(selected_evader.ID) 

            e2a = []
            p2a = []

## capture distance for pursuers and evader.
def capture_dist (p,e,d):
     dist = (p.x - e.x)**2 + (p.y- e.y)**2
     return dist <= d

def capture_dist2 (p,e):
    dist = (p.x - e.x)**2 + (p.y- e.y)**2
    return dist

## helper function to find minimum cost(distance) using the hungarian linear assignment algorithm 
## Adapted from (Algorithm 2 -Zang )
## https://arxiv.org/pdf/2103.15660.pdf

def hungarian_lap(P,E):
    cost_matrix =  np.zeros((n,m))
    for ii in range(n):
        for jj in range(m):
            cost_matrix[ii][jj] = capture_dist2(P[ii],E[jj])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    return row_ind,col_ind

### Helper to find reachability set 

def forward_reachable_set(agent, x_field, y_field, v_field, u_field, dt, t):

    # initialization 
    
    iterations = 15

    x_phi, y_phi = np.meshgrid(np.linspace(-dim,dim,2*dim+1),np.linspace(-dim,dim,2*dim+1))
    x_phi = x_phi - agent.x 
    y_phi = y_phi - agent.y 
    phi = np.sqrt(x_phi**2 + y_phi**2)
    
    # forward propagation
    for k in range(iterations):
        dphi = np.gradient(phi)
        x_dphi, y_dphi = np.gradient(phi)
        dphi_norm = np.sqrt(np.sum(np.square(dphi),axis=0))

        field_dphi = np.zeros([2*dim+1,2*dim+1])
        for i in range(2*dim+1):
            for j in range(2*dim+1):
                a = np.array([v_field[i,j], u_field[i,j]])
                b = np.array([x_dphi[i,j], y_dphi[i,j]])
                field_dphi[i,j] = a @ b

        phi = phi - dt*F*dphi_norm - field_dphi *dt

        #plt.quiver(x_field,y_field,x_phi, y_phi)
        #cs = plt.contour(x_field,y_field,phi,levels=[-2, 0, 2],colors=['#808080', '#A0A0A0', '#C0C0C0'], extend='both')
        #cs.cmap.set_over('red')
        #cs.cmap.set_under('blue')
        #cs.changed()
        
        plt.quiver(x_field,y_field,u_field,v_field)
        plt.scatter(agent.x,agent.y)
        plt.contour(x_field,y_field,phi,0)
        #plt.title(k)
        plt.clf()
        #plt.show()

    return phi, dphi_norm, x_dphi, y_dphi


def is_within_reachable(e, x_field,y_field, phi):
    
    coordinates = np.where(phi<=0.1)
    cx, cy = coordinates
    cx = cx - dim
    cy = cy - dim
    plt.contour(x_field,y_field,phi,0)
    plt.scatter(cx,cy)
    #plt.show()
    return e.x in cx and e.y in cy

def toc(phi, dphi_norm, x_dphi, y_dphi):
    scaling_factor = 0.52
    return F * x_dphi, F* y_dphi # @ np.linalg.inv(dphi_norm)

def time_to_capture(p,e):
    ## Assumed they moved in same direction/ flow field direction
    dist = capture_dist2(p,e)
    mag_velocity_purser = math.sqrt ((p.vx)**2 + (p.vy)**2)
    vel_ev = e.vel()
    mag_velocity_evader = math.sqrt((vel_ev[0])**2 + (vel_ev[0])**2)
    time = dist/ abs(mag_velocity_evader - mag_velocity_purser)

    return time

def time_reachability(P,E):
    time_matrix =  np.zeros((n,m))
    for ii in range(n):
        for jj in range(m):
            time_matrix[ii][jj] = time_to_capture(P[ii],E[jj])
    return time_matrix


def play_game():
    flowfield_mode = 'ON' # flowfield mode: 'ON' or 'OFF'
    task_assign_mode = 'HUNGARIAN' # task assignment mode: 'ZAVLANOS' or 'HUNGARIAN' 
    display_game = True #True

    print("Playing...")

    pygame.init()
    clock = pygame.time.Clock()

    if display_game:
        # Set up display
        # https://www.pygame.org/docs/tut/PygameIntro.html
        size = width, height = 600, 600 # display size 500 x 500 px
        screen = pygame.display.set_mode(size)
        scale = width / 60
        font = pygame.font.SysFont(None, 100)
        # Wait 3 seconds (for screen recording)
        time.sleep(3)

    # Initialize pursuers, evaders
    # Field is 6m x 6m, centered at the origin
    # Pursuers start in box [-2, -2] to [-1, -1]
    # Evaders start on a circle of radius R
    P = []
    for ii in range(n):
        x = random.random() - 2.0
        y = random.random() - 2.0
        p = Pursuer(x, y,ii)
        P.append(p)

    E = []
    R = 2.0
    for ii in range(m):
        a = (2 * math.pi / m) * ii
        x = R * np.cos(a)
        y = R * np.sin(a)
        e = Evader(x, y, ii)
        E.append(e)

    dt = 1

    

    function = lambda x, y: (-sin(y), x)
    h = VectorField(function,int(dim/2), 0.012,50)



    
    # setting up flow field
    #u, v = vel_form(cen_list, dim, ratio_list)
    if flowfield_mode == 'ON':
        #x_field, y_field, u_field, v_field = circular_velocity_field(int(dim/2), 2.0)
        x_field, y_field, u_field, v_field = circular_velocity_field(0.5)
    else:
        # x_field, y_field, u_field, v_field = circular_velocity_field(int(dim/2), 0.0)
        x_field, y_field, u_field, v_field = circular_velocity_field(0.0)

    
    # use u and v to update position of purusers and evaders. 
    #plt.quiver(x_field,y_field,u_field,v_field)
    #plt.show()
    #breakpoint()

    is_game_over = False
    
    t0 = 0.0
    t = t0

    while not is_game_over:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Assign pursuer-evader pairs
        if task_assign_mode == 'ZAVLANOS':
            task_assignment(P,E)
            A = []
            for p_ind in range(n):
                if len(P[p_ind].I_a) == 1:
                    A.append((p_ind, P[p_ind].I_a[0]))
        elif task_assign_mode == 'HUNGARIAN':
            p,e = hungarian_lap(P,E)
            A = list(zip(p,e))
        else:
            time_matrix = time_reachability(P,E)
            ###%%%% TODO: MAKE ASSIGNMENTS WITH THIS MATRIX %%%%###

        # Check assignments
        '''
        ii =  1
        for pursuer in P:
            print("Pursuer ", ii ," location: ", pursuer.x, pursuer.y, "Pursuer Assignment",pursuer.I_a, pursuer.I_t )
            ii = ii + 1
        ii =  1
        for evader in E:
            print("Evader ", ii ," location: ", evader.x, evader.y, "Pursuer Assignment",evader.ID)
            ii = ii + 1
        '''

        # Integrate dynamics
        for p_ind, e_ind in A:
            e = E[e_ind]
            # Compute optimal velocity using Zavlanos method
            #vx, vy = P[p_ind].vel(e.x, e.y, t, t0) 
            
            # Add in flowfield velocities
            print("pursuer's location:", P[p_ind].x, P[p_ind].y)
            vx = v_field[int(P[p_ind].x + dim), int(P[p_ind].y + dim)] 
            vy = u_field[int(P[p_ind].x + dim), int(P[p_ind].y + dim)]
            
            
            # Compute optimal velocity from HJ equation if within reachability set
            #phi, dphi_norm, x_dphi, y_dphi = forward_reachable_set(P[p_ind], x_field, y_field, v_field, u_field, dt, t)
            #if is_within_reachable(E[e_ind], x_field,y_field, phi):
            #    vx, vy = toc(phi, dphi_norm, x_dphi, y_dphi)

            P[p_ind].vx = vx
            P[p_ind].vy = vy
            P[p_ind].move(vx, vy, dt)

        for ii in range(m):
            if flowfield_mode == 'OFF':
                vx, vy = E[ii].vel() # move along a circle
                E[ii].move(vx, vy, dt)
            else:
                #breakpoint()
                print("evader's location:", E[ii].x, E[ii].y)
                E[ii].move(v_field[int(E[ii].x + dim), int(E[ii].y + dim)], u_field[int(E[ii].x + dim), int(E[ii].y + dim)], dt)
                
                # something like: E[ii].move(vx + v_field[E[ii].x], vy + u_field[E[ii].y], dt)
                # check if the dimensions of velocity field is same as dimensions (and range of axis) to the grid for pursuers and evaders that Kristen coded

        # Visualize
        if display_game:
            screen.fill((255,255,255))
            h.draw(screen)
            [p.display(screen, scale, width) for p in P]
            [e.display(screen, scale, width) for e in E]
            clock.tick(30)
            pygame.display.update()

        v= []
        if len(A) == n:
             for i in A:
                if (True == capture_dist(P[i[0]],E[i[1]],0.01)):
                      v.append(i[0])
        if len(v) == n:
             is_game_over = True
             print('--- GAME SUMMARY ---')
             print('Flowfield: ' + flowfield_mode)
             print('Task assignment: ' + task_assign_mode)
             print('Time to capture: ' + str(t))

        # Current time
        t = t + dt
   

if __name__ == "__main__":
	play_game()