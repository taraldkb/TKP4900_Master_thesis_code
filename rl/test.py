import torch
import json
from environment.water_injection_env import WaterInjectionEnv
from environment.cfd_interface import run_cfd_step
from utils.map_value_function import *
from utils.read_report_function import *
import matplotlib.pyplot as plt
from rl.network import PolicyNet


# read config
with open("configs/RL_config.json", "r") as f:
    CONFIG = json.load(f)


# create tester
def test_agent(policy_path=CONFIG["save_path"]):
    env = WaterInjectionEnv(run_cfd_step)
    policy = PolicyNet(CONFIG["state_dim"], CONFIG["action_dim"], CONFIG["hidden_size"])
    policy.load_state_dict(torch.load(policy_path))
    policy.eval()

    time_step = list(range(0, 23, 2))
    wind_profiles_lib = [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                         [0.5, 0.95, 0.95, 0.95, 0.95, 0.2, 0.2, 0.2, 0.2, 0.2],
                         [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 0.8]]
    sp_profiles_lib = [[20.0, 25.0, 25.0, 25.0, 30.0, 30.0, 10.0, 10.0, 10.0, 10.0],
                       [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
                       [20.0, 25.0, 30.0, 30.0,  30.0, 30.0, 30.0, 30.0, 30.0, 30.0]]

    for ep in range(3):
        # get variable profile
        wind_profile = wind_profiles_lib[ep]
        sp_profile = sp_profiles_lib[ep]

        # reset state
        state = env.testing_reset()

        # save for plotting
        injection1 = [map_value(0.25, 0, 20)]
        injection2 = [map_value(0.25, 0, 20)]
        mass = [map_value(0.5, 0, 100)]
        conc_plot = [[] for _ in range(8)]
        wind_plot = [map_value(state[-2], 0.0, 20.0)]
        sp_plot = [state[-1]]

        for i in range(len(conc_plot)):
            conc_plot[i].append(state[i])

        total_reward = 0
        total_rewards = []
        rewards = []
        done = False
        counter = 0

        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32)
                mean, _ = policy(state_tensor)
                action = mean.numpy()

                # save actions for plotting
                injection1.append(map_value(action[0], 0, 20))
                injection2.append(map_value(action[1], 0, 20))
                mass.append((map_value(action[2], 0, 100)))

            state, reward, done, _ = env.test_step(action, wind_profile[counter], sp_profile[counter])
            total_reward += reward

            # save reward and state for plotting
            total_rewards.append(total_reward)
            rewards.append(reward)
            wind_plot.append(map_value(state[-2], 0.0, 20.0))
            sp_plot.append(state[-1])
            for i in range(len(conc_plot)):
                conc_plot[i].append(state[i])
            counter += 1

        print(f"[Test] Episode {ep + 1}: reward = {total_reward:.2f}")

        # add duplicate for plotting visuals
        injection1.append(injection1[-1])
        injection2.append(injection2[-1])
        mass.append(mass[-1])
        wind_plot.append(wind_plot[-1])
        sp_plot.append(sp_plot[-1])
        for i in range(len(conc_plot)):
            conc_plot[i].append(conc_plot[i][-1])

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.step(time_step, injection1, where="post", label="Injection 1")
        plt.step(time_step, injection2, where="post", label="Injection 2")
        plt.legend()
        plt.grid()
        plt.xlabel("Time [s]")
        plt.ylabel("Velocity [m/s]")
        plt.title(f"Injection velocity test run {ep+1}")
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.step(time_step, mass, where="post")
        plt.grid()
        plt.xlabel("Time [s]")
        plt.ylabel("Mass flow [kg/s]")
        plt.title(f"Injection mass flow test run {ep+1}")
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.step(time_step, wind_plot, where="post")
        plt.grid()
        plt.xlabel("Time [s]")
        plt.ylabel("Velocity [m/s]")
        plt.title(f"Wind velocity test run {ep+1}")
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.step(time_step, sp_plot, where="post", label="Concentration setpoint")
        for i in range(4):
            plt.step(time_step, conc_plot[i], where="post", label=f"Zone {i+1}")
        plt.grid()
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel(r"Concentration [kg/m$^3$]")
        plt.title(f"Concentrations left zones and Setpoint test run {ep+1}")
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.step(time_step, sp_plot, where="post", label="Concentration setpoint")
        for i in range(4, 8):
            plt.step(time_step, conc_plot[i], where="post", label=f"Zone {i + 1}")
        plt.grid()
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel(r"Concentration [kg/m$^3$]")
        plt.title(f"Concentrations right zones and Setpoint test run {ep+1}")
        plt.show()

        plot_water("water_loss.out", ep+1)