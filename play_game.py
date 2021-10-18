import numpy as np
import assign_pairs

# Helper functions
def is_neighbor(x_i, x_j) # for pursuer-pursuer

def is_within_reach(x_i, y_k) # for pursuer-evader

def tb(list(int)) #input: list of pursuers up for tie break
    # pick one of the pursuers randomly

def play_game():
    print("Playing...")
    n = 4 # number of pursuers
    m = 4 # number of evaders
    
    # list of initial positions (X = pursuers, Y = evaders)
    X = # [tuple(float, float), ...]
    Y = # [tuple(float, float), ...]
    
    dt = 0.01
    
    is_game_over = False
    
    while not is_game_over:
        # Assign pursuer-evader pairs
        # Output: A = [(int: P, int: E), ...]
        # I = list of pairs [(I^a(list: int), I^t(list: int)), ...]
        # Ex:
        # I[1][1]: pursuer 1's I^a list
        # I[4][2]: pursuer 4's I^t list 
        
        # Control inputs u are computed
        Ui = [tuple(float, float)] # 2D control inputs for pursuers
        Uk = [tuple(float, float)] # 2D control inputs for evaders
        
        # Integrate dynamics (ie x(t+dt) = x(t) + dt * u(t))
        
        # Visualize
        
        
    

if __name__ == "__main__":
	play_game()