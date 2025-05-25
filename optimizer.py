import numpy as np
from pyomo.environ import (
    ConcreteModel, Var, Objective, Constraint, SolverFactory, Binary,
    Param, value, RangeSet, minimize, Expr_if
)
from rl.optimazation_test import pi
from utils.cleanup import cleanup

# --- Constants ---
h = [50, 75, 100]  # height options
Q_m = 1
Q_u = 0.1
state = [13.99, 14.04, 6.89, 21.62, 4.81, 26.73, 14.65, 2.90, 0, 0]

# Clean up any previous simulation output
cleanup()

# --- Precompute pi(state, H, N) ---
pi_lookup = {}
for i, height in enumerate(h):
    for y_val in [0, 1]:  # y = 0 or 1 → N = y + 1
        N = y_val + 1
        pi_val = pi(state, height, N)
        #print(f"mass flow for H= {height}, N = {N} is {pi_val}")
        pi_lookup[(i, y_val)] = pi_val

# --- Pyomo Model ---
model = ConcreteModel()

# Sets
model.z_index = RangeSet(0, len(h) - 1)  # {0, 1, 2}
model.y_index = RangeSet(0, 1)           # {0, 1}

# Decision variables
model.z = Var(model.z_index, domain=Binary)     # select height
model.y = Var(domain=Binary)                    # injector count binary
model.w = Var(model.z_index, domain=Binary)     # helper for z[i] * y

# Parameters
model.pi = Param(model.z_index, model.y_index, initialize=pi_lookup, mutable=True)

# --- Constraints ---
# Only one height (z[i]) can be chosen
def z_constraint_rule(model):
    return sum(model.z[i] for i in model.z_index) == 1
model.z_constraint = Constraint(rule=z_constraint_rule)

# Linearization constraints: w[i] = z[i] * y
def w_constraint1(model, i):
    return model.w[i] <= model.z[i]

def w_constraint2(model, i):
    return model.w[i] <= model.y

def w_constraint3(model, i):
    return model.w[i] >= model.z[i] + model.y - 1

model.w_con1 = Constraint(model.z_index, rule=w_constraint1)
model.w_con2 = Constraint(model.z_index, rule=w_constraint2)
model.w_con3 = Constraint(model.z_index, rule=w_constraint3)

# --- Objective ---
def objective_rule(model):
    H = sum(model.z[i] * h[i] for i in model.z_index)
    N = model.y + 1

    # Linear pi selection: pi[i, y] = pi[i, 0] * (1 - y) + pi[i, 1] * y
    # Equivalent using z[i], w[i] to avoid nonlinear terms
    pi_expr = sum(
        model.z[i] * model.pi[i, 0] +
        model.w[i] * (model.pi[i, 1] - model.pi[i, 0])
        for i in model.z_index
    )

    cost_m = H + N * 50
    return Q_m * cost_m + Q_u * pi_expr

model.obj = Objective(rule=objective_rule, sense=minimize)

# --- Solve the model ---
solver = SolverFactory('glpk')
result = solver.solve(model)

# --- Extract results ---
H_opt = value(sum(model.z[i] * h[i] for i in model.z_index))
N_opt = value(model.y) + 1
pi_opt = sum(
    value(model.z[i]) * pi_lookup[(i, int(value(model.y)))]
    for i in model.z_index
)
cost_m_opt = H_opt + N_opt * 50
total_cost = Q_m * cost_m_opt + Q_u * pi_opt

# --- Print results ---
print("\nOptimization complete")
print(f"Optimal Height: {H_opt} mm")
print(f"Optimal Number of Injectors: {N_opt}")
print(f"PI Value (mass flow): {pi_opt}")
print(f"Total Cost: {total_cost}")







