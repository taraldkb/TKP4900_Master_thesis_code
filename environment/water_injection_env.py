import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import json
import ansys.fluent.core as pyfluent
from ansys.fluent.core import launch_fluent
import random
import time


class WaterInjectionEnv(gym.Env):
    def __init__(self, run_cfd_step_fn, config_path="configs/design_settings.json",
                 case_dir="cases", trans_controls_path="configs/transient_control_settings.json",
                 report_path="concentration.out", loss_path="water_loss.out",
                 water_path="water_usage.out"):
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
        self.loss_report_path = loss_path
        self.water_usage_report_path = water_path
        self.run_cfd_step = run_cfd_step_fn

        # create simulation variables
        self.fluent_session = None
        self.max_steps = 10
        self.step_count = 0
        self.time_step_type = self.trans_params["type"]
        self.iter_per_timestep = self.trans_params["ipt"]
        self.time_step_size = self.trans_params["step_size"]
        self.time_step_total = self.trans_params["total_steps"]
        self.state = None

        # set variable values
        self.wind_profile_lib = [
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75, 0.75],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25],
            [0.5, 0.5, 0.75, 0.75, 0.5, 0.5, 0.75, 0.75, 0.85, 0.9],
            [0.5, 0.25, 0.5, 0.75, 0.5, 0.5, 0.25, 0.5, 0.75, 0.5,]
        ]
        self.sp_profile_lib = [
            [20, 20, 20, 20, 20, 15, 15, 15, 15, 15],
            [20, 20, 20, 20, 20, 30, 30, 30, 30, 30],
            [20, 15, 10, 9, 8, 5, 5, 10, 10, 15],
            [20, 20, 10, 10, 10, 10, 30, 30, 30, 30]
        ]
        self.wind_profile = None
        self.sp_profile = None

        self._current_wind = 0.5
        self.setpoint = 20.0


        # create initall state and actions for intializing system
        self.initial_actions = np.array([0.25, 0.25, 0.49494949])
        self.state = self.state = np.concatenate([np.full(8, 0.0), [self._current_wind], [self.setpoint]])

        # create action space 3 actions [injection1, injection2, mass_flow]
        self.action_space = spaces.Box(low=0.0,
                                       high=1.0,
                                       shape=(3,), dtype=np.float32)

        # create observation space 10 obs [8 zones, wind, setpoint], value between 0-1, set proper values for zone value
        self.observation_space = spaces.Box(low=np.zeros(10),
                                            high=np.append(np.full(8, 100), [1.0, 50]),
                                            shape=(10,), dtype=np.float32)

    def reset(self):  # reset environment for new episode
        self._start_fluent_with_case()
        self._current_wind = 0.5
        self.setpoint = 20.0
        self.step_count = 0
        self._get_profiles()
        self._get_initial_state()

        return self.state

    # create function for taking time step
    def step(self, action):

        # check for wind update
        self._update_variables()

        # Run cfd and gather observation
        next_state, water_loss = self.run_cfd_step(
            self.fluent_session,
            self.state,
            action,
            self.design_params,
            self.report_path,
            self.loss_report_path
        )

        self.state = np.concatenate([next_state, [self.setpoint]])
        reward = self._compute_reward(self.state, action, water_loss)
        self.step_count += 1
        done = self.step_count >= self.max_steps

        return self.state, reward, done, {}

    def _update_variables(self):
        wind = self.wind_profile[self.step_count]
        self.setpoint = self.sp_profile[self.step_count]

        self.state[-2] = wind
        self.state[-1] = self.setpoint

    def _compute_reward(self, state, action, water_loss):
        # reward function for RL, should check this reward
        moisture = state[:-2]
        sp = state[-1]
        return -np.sum((moisture - sp) ** 2) - 0.01 * np.sum(action ** 2) - 0.1*water_loss

    def _start_fluent_with_case(self):  # start fluent solver with correct case setup

        # remove active sessions
        if self.fluent_session is not None:
            self.fluent_session.exit()
            self.fluent_session = None

        # clean up files between sessions
        if os.path.exists(self.report_path):
            for _ in range(10):
                try:
                    os.remove(self.report_path)
                    break
                except PermissionError:
                    time.sleep(0.5)

        if os.path.exists(self.water_usage_report_path):
            for _ in range(10):
                try:
                    os.remove(self.water_usage_report_path)
                    break
                except PermissionError:
                    time.sleep(0.5)
        if os.path.exists(self.loss_report_path):
            for _ in range(10):
                try:
                    os.remove(self.loss_report_path)
                    break
                except PermissionError:
                    time.sleep(0.5)

        case_file = f"N{self.N}_H{self.H}.cas.h5"
        case_path = os.path.join(self.case_dir, case_file)

        if not os.path.exists(case_path):  # check for existing file if not raise error
            raise FileNotFoundError(f"Fluent case file not found: {case_path}")

        # create fluent session and load correct case
        self.fluent_session = launch_fluent(mode="solver", precision="double", processor_count=8,
                                            dimension=pyfluent.Dimension.TWO)

        self.fluent_session.file.read(file_type="case", file_name=case_path)

        # set transient simulation controls
        trans_controls = self.fluent_session.solution.run_calculation.transient_controls
        trans_controls.type = self.time_step_type
        trans_controls.max_iter_per_time_step = self.iter_per_timestep
        trans_controls.time_step_size = self.time_step_size
        trans_controls.time_step_count = self.time_step_total

        # initialize case with hybrid initialization
        self.fluent_session.solution.initialization.hybrid_initialize()

    def _get_initial_state(self):
        initial_state, _ = self.run_cfd_step(
            self.fluent_session,
            self.state,
            self.initial_actions,
            self.design_params,
            self.report_path,
            self.loss_report_path
        )

        self.state = np.concatenate([initial_state, [self.setpoint]])

    def _get_profiles(self):
        rand_int_wind = random.randint(0, 3)
        rand_int_sp = random.randint(0, 3)

        self.wind_profile = self.wind_profile_lib[rand_int_wind]
        self.sp_profile = self.sp_profile_lib[rand_int_sp]

