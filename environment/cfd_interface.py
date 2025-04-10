import numpy as np
import time
import os
from utils.read_report_function import *
from utils.map_value_function import *


def run_cfd_step(solver, current_state, action, design_params, report_path, water_report_path):

    """
    Run a step in cfd model, 2 seconds/ 20 time steps
    :param water_report_path: path to water_loss.out file string
    :param solver: Active pyfluent session object
    :param current_state: current environment state array size (9, ) [concentrationX8, wind_velocity, setpoint]
    :param action: MV values array size (3, ) [speed1, speed2, mass_flow]
    :param design_params: design parameteres hight H and injection points N {'N': 1 or 2, 'H': 50/75/100}
    :param report_path: path to report.out file string
    :return next_state: state after cfd step array size (9, ) [concentrationX8, wind_velocity]
    """

    # Unpack state and manipulated variables
    wind_velocity = current_state[-2]
    speed1, speed2, mass_flow = action
    N = design_params["N"]

    # Set boundary conditions
    try:
        solver.setup.boundary_conditions.velocity_inlet["wind"].momentum.value = map_value(wind_velocity(1.0, 20.0))

    except Exception as e:
        print(f"[WARNING] Could not set wind velocity: {e}")

    # set injection velocity
    try:
        injection1 = solver.setup.models.discrete_phase.injections["injection1"]
        injection1.initial_values.velocity.magnitude = map_value(speed1, 0.0, 20.0)
        injection1.initial_values.mass_flow_rate.total_flow_rate = map_value(mass_flow, 1.0, 100.0)
    except Exception as e:
        print(f"[WARNING] Could not update injection1: {e}")

    if N == 2:
        try:
            injection2 = solver.setup.models.discrete_phase.injections["injection2"]
            injection2.initial_values.velocity.magnitude = map_value(speed2, 0.0, 20.0)
            injection2.initial_values.mass_flow_rate.total_flow_rate = map_value(mass_flow, 1.0, 100.0)
        except Exception as e:
            print(f"[WARNING] Could not update injection2: {e}")

    # run simulation

    solver.solution.run_calculation.calculate()

    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Report file could not be could: {report_path} ")

    # read reports
    next_state = read_concentrations(report_path)
    next_state.append(wind_velocity)

    water_loss = read_single_data_file(water_report_path)

    return next_state, water_loss







