import numpy as np
from scipy.optimize import minimize
from rl.optimazation_test import pi
from optimazation.get_state_function import get_state


# define material Cost function
def C_m(H, N):
    return H*2 + N*30


# define objective function
def C_p(cost_m, u, q_m=1, q_u=1):
    return q_m*cost_m + q_u*u


# define optimization problem
def problem(vars, x, q_m=1, q_u=1):
    # define constants
    h = np.array([50, 75, 100])
    n_z = len(h)

    # unpack decision variables
    z = vars[:n_z]
    y = vars[-1]

    # add variable definitions
    H = np.dot[z, h]
    N = y+1
    Cost_M = C_m(H, N)
    u = pi(x, H, N)
    return C_p(Cost_M, u, q_m, q_u)







