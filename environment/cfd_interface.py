import numpy as np
import time
import os
from utils.read_report_function import *
from utils.map_value_function import *

def run_cfd_step(fluent_session, current_state, action, design_params, report_path):

    """
    Run a step in cfd model, 2 seconds/ 20 time steps
    :param fluent_session: Active pyfluent object
    :param current_state: current environment state array size (9, ) [concentrationX8, wind_velocity]
    :param action: MV values array size (3, ) [speed1, speed2, mass_flow]
    :param design_params: design parameteres hight H and injection points N {'N': 1 or 2, 'H': 50/75/100}
    :param report_path: path to report.out file string
    :return next_state: state after cfd step array size (9, ) [concentrationX8, wind_velocity]
    """

    solver = fluent_session.solver

    # Unpack state and manipulated variables
    wind_velocity = current_state[-1]
    speed1, speed2, mass_flow = action
    N = design_params["N"]

    # Set boundary conditions
    try:
        solver.setup.boundary_conditions.velocity_inlet["wind"].vmag = wind_velocity

    except Exception as e:
        print(f"[WARNING] Could not set wind velocity: {e}")

    # set injection velocity
    try:
        injection1 = solver.setup.models.discrete_phase.injections["injection1"]
        injection1.properties.velocity = map_value(speed1, 0, 20)
        injection1.properties.mass_flow_rate = map_value(mass_flow, 50, 500)
    except Exception as e:
        print(f"[WARNING] Could not update injection1: {e}")

    if N == 2:
        try:
            injection2 = solver.setup.models.discrete_phase.injections["injection1"]
            injection2.properties.velocity = map_value(speed2, 0, 20)
            injection2.properties.mass_flow_rate = map_value(mass_flow, 50, 500)
        except Exception as e:
            print(f"[WARNING] Could not update injection2: {e}")

    # run simulation

    solver.solution.run_calculation.calculate()

    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Report file could not be could: {report_path} ")

    # read report
    next_state = read_report(report_path)

    return next_state







