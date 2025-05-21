import numpy as np
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory, Binary, summation, value
from rl.optimazation_test import pi
from optimazation.get_state_function import get_state


# define constants
h = [50, 75, 100]
Q_m = 1
Q_u = 1

# get state from CFD
state = get_state()


# define material Cost function
def C_m(H, N):
    return H*2 + N*30


# define objective function
def C_p(cost_m, u, q_m=1, q_u=1):
    return q_m*cost_m + q_u*u


# create pymo model
model = ConcreteModel()

# define the decision variables
model.z_index = range(len(h))
model.z = Var(model.z_index, domain=Binary)
model.y = Var(domain=Binary)


# define constraint on z variable and add to model add constraint to model
def z_rule(model):
    sum_z = 0
    for i in model.z_index:
        sum_z += model.z[i]
    return sum_z


model.z_constraint = Constraint(rule=z_rule)


# define variable relations
def H_rel(model):
    sum_H = 0
    for i in model.z_index:
        model.z[i]*h[i]
    return sum_H


def N_rel(model):
    return model.y + 1


# create optimization problem
def optim_rule(model):
    H = H_rel(model)
    N = N_rel(model)
    u = pi(state, H, N)
    cost_m = C_m(H, N)
    return C_p(cost_m, u, Q_m, Q_u)


model.obj =Objective(rule= optim_rule, sense=1)

solver = SolverFactory('glpk')
result = solver.solve(model)

H_opt = value(H_rel(model))
N_opt = value(N_rel(model))

print("Optimization complete")
print(f"Optimal Height: {H} millimeters")
print(f"Optimal amount of injectors: {N} injector(s)")









