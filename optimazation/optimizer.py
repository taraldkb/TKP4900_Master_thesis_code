import numpy as np
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, \
    SolverFactory, Binary, summation, value
from rl.optimazation_test import pi
from optimazation.get_state_function import get_state
from utils.cleanup import cleanup


# define constants
h = [50, 75, 100]
Q_m = 1
Q_u = 1
N_cost = 50


# get state from CFD
state = get_state()

# remove report files
cleanup()










