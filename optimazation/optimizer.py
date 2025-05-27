import numpy as np
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, \
    SolverFactory, Binary, value, RangeSet, Param, minimize
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

# create a lookup table for precomputed policy values
pi_lookup = {}
for i, height in enumerate(h):
    for y_val in [0, 1]:
        N = y_val + 1
        pi_val = pi(state, height, N)
        pi_lookup[(i, y_val)] = pi_val

# create pyomo model
model = ConcreteModel()

# create set index for y and z
model.z_index = RangeSet(0, len(h)-1)
model.y_index = RangeSet(0, 1)

# create decision variables
model.z = Var(model.z_index, domain=Binary)
model.y = Var(domain=Binary)

# helper decision variable to keep lookup table binary
model.w = Var(model.z_index, domain=Binary)

# create parameters
model.pi = Param(model.z_index,
                 model.y_index,
                 initialize=pi_lookup,
                 mutable=True
                 )

# create constraints


# only one z_i non zero
def z_constraint_rule(model):
    z_sum = 0
    for i in model.z_index:
        z_sum += model.z[i]
    return z_sum == 1


# add constraint to model
model.z_constraint = Constraint(rule=z_constraint_rule)


# linearization constraints
def w_constraint1(model, i):
    return model.w[i] <= model.z[i]


def w_constraint2(model, i):
    return model.w[i] <= model.y


def w_constraint3(model, i):
    return model.w[i] >= model.z[i] + model.y - 1


# add constraints to model
model.w_con1 = Constraint(model.z_index, rule=w_constraint1)
model.w_con2 = Constraint(model.z_index, rule=w_constraint2)
model.w_con3 = Constraint(model.z_index, rule=w_constraint3)


# create objective function
def objective_rule(model):
    H = sum(model.z[i]*h[i] for i in model.z_index)
    N = model.y + 1

    # select correct value for lookup tabel
    pi_expr = sum(
        model.z[i] * model.pi[i, 0] +
        model.w[i] * (model.pi[i, 1] - model.pi[i, 0])
        for i in model.z_index
    )

    cost_m = H + N * N_cost
    return Q_m * cost_m + Q_u * pi_expr


# add objective function to model
model.obj = Objective(rule=objective_rule, sense=minimize)

# solve model
solver = SolverFactory('glpk')
result = solver.solve(model)


# extract results
H_opt = value(sum(model.z[i] * h[i] for i in model.z_index))
N_opt = value(model.y) + 1
pi_opt = sum(
    value(model.z[i]) * pi_lookup[(i, int(value(model.y)))]
    for i in model.z_index
)
cost_m_opt = H_opt + N_opt * N_cost
total_cost = Q_m * cost_m_opt + Q_u * pi_opt


print(f"Optimal H: {H_opt}")
print(f"Optimal N: {N_opt}")
print(f"Total cost: {total_cost}")






