import numpy as np
import math
import assign_pairs
import numpy as np

# Helper functions
def is_neighbor(x_i, x_j) # for pursuer-pursuer

def is_within_reach(x_i, y_k) # for pursuer-evader
    abs(x_i-y_k)^2 < δ 

def is_taken(y_k, I)
    return size(i[y_k][1])==1

def tb(list(int)) #input: list of pursuers up for tie break
    # pick one of the pursuers randomly
 
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
        A = [(-1,-1) for i in range(n)]
        #I = [([i for i in range(m)],[-1 for i in range(m)]) for j in range(n)]
        I = np.empty((n,),dtype=object)
        for i,v in enumerate(I): 
            I[i]=[[k for k in range(m)],v]
        

        l=0
        for v in I: 
            v.append(l)
            l+=1
        task_assigment(I, A)
        # Control inputs u are computed
        # Ui = [tuple(float, float)] # 2D control inputs for pursuers
        # Uk = [tuple(float, float)] # 2D control inputs for evaders
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