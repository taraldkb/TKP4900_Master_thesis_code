import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import json
import ansys.fluent.core as pyfluent
from ansys.fluent.core import launch_fluent
import random


class WaterInjectionEnv(gym.Env):
    def __init__(self, run_cfd_step_fn, config_path="configs/design_settings.json",
                 case_dir="cases", trans_controls_path="configs/transient_control_settings.json",
                 report_path="results/report.out"):
        super().__init__()

        # load config
        with open(config_path, "r") as f:
            self.design_params = json.load(f)
        # load transient control
        with open(trans_controls_path, "r") as f:
            self.trans_params = json.load(f)

        # initial parameters and directories
        self.N = self.design_params["N"]
        self.H = self.design_params["H"]
        self.case_dir = case_dir
        self.report_path = report_path
        self.run_cfd_step = run_cfd_step_fn
        self.time_step_type = self.trans_params["type"]
        self.iter_per_timestep = self.trans_params["ipt"]
        self.time_step_size = self.trans_params["step_size"]
        self.time_step_total = self.trans_params["total_steps"]

        # create simulation variables
        self.fluent_session = None
        self.max_steps = 20
        self.step_count = 0
        self.state = None

        # create wind tracking parameters
        self.wind_change_interval = 5
        self._wind_step_counter = 0
        self._current_wind = 0.5

        # create action space 3 actions [injection1, injection2, mass_flow]
        self.action_space = spaces.Box(low=0.0,
                                       high=1.0,
                                       shape=(3,), dtype=np.float32)

        # create observation space 8 obs [8 zones, wind], value between 0-1, set propper values for zone value
        self.observation_space = spaces.Box(low=np.zeros(9),
                                            high=np.append(np.full(8, 15.0), np.float32(15.0)),
                                            shape=(9,), dtype=np.float32)

        # set points, static for now CHANGE LATER !!!!
        self.setpoints = np.full(8, 0.6)

    def reset(self):  # reset environment for new episode
        self._start_fluent_with_case()
        self.step_count = 0
        self._wind_step_counter = 0

        # create initial state MIGHT NEED TO CHANGE [ dpm concentration X8, wind velocity]
        self._current_wind = 0.5
        self.state = np.concatenate([np.full(8, 0.0), [self._current_wind]])
        return self.state

    # create function for taking time step
    def step(self, action):

        # check for wind update
        self._update_wind_velocity()

        # Run cfd and gather observation
        next_state = self.run_cfd_step(
            self.fluent_session,
            self.state,
            action,
            self.design_params,
            self.report_path
        )

        reward = self._compute_reward(next_state, action)
        self.state = next_state
        self.step_count += 1
        done = self.step_count >= self.max_steps

        return self.state, reward, done, {}

    def _update_wind_velocity(self):
        if self._wind_step_counter % self.wind_change_interval == 5 and self._wind_step_counter != 0:

            # update with new wind after 5 steps
            self._current_wind = random.uniform(0.0, 1.0)

        # update counter and value
        self.state[-1] = self._current_wind
        self._wind_step_counter += 1


    def _compute_reward(self, state, action):
        # reward function for RL, should check this reward
        moisture = state[:-1]
        return -np.sum((moisture - self.setpoints) ** 2) - 0.01 * np.sum(action ** 2)

    def _start_fluent_with_case(self):  # start fluent solver with correct case setup
        case_file = f"N{self.N}_H{self.H}.cas.h5"
        case_path = os.path.join(self.case_dir, case_file)

        if not os.path.exists(case_path):  # check for existing file if not raise error
            raise FileNotFoundError(f"Fluent case file not found: {case_path}")

        # create fluent session and load correct case
        self.fluent_session = launch_fluent(mode="solver", precision="double", processor_count=8,
                                            dimension=pyfluent.Dimension.TWO)
        solver = self.fluent_session.solver
        solver.file.read(file_type="case", file_name=case_path)

        # set transient simulation controls
        trans_controls = solver.solution.run_calculation.transient_controls
        trans_controls.type = self.time_step_type
        trans_controls.max_iter_per_time_step = self.iter_per_timestep
        trans_controls.time_step_size = self.time_step_size
        trans_controls.time_step_count = self.time_step_total

        # initialize case with hybrid initialization
        solver.solution.initialization.hybrid_initialize()
