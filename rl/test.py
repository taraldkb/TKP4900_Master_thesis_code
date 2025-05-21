import torch
import json
from environment.water_injection_env import WaterInjectionEnv
from environment.cfd_interface import run_cfd_step
from utils.map_value_function import *
import matplotlib.pyplot as plt
from rl.network import PolicyNet
import matplotlib as mpl

# nice plotting
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "lines.linewidth": 1.5,
    "legend.frameon": False,
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

fig_width = 6.27
fig_height = fig_width * 0.618

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
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.step(time_step, injection1, where="post", label="Injection 1")
        ax.step(time_step, injection2, where="post", label="Injection 2")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Velocity [m/s]")
        ax.set_title(f"Injection velocity test run {ep+1}")
        ax.legend()
        ax.grid()
        fig.tight_layout()
        fig.savefig(f"injection_vel_run{ep+1}.pdf")
        plt.show()

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.step(time_step, mass, where="post")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Mass flow [kg/s]")
        ax.set_title(f"Injection mass flow test run {ep+1}")
        ax.grid()
        fig.tight_layout()
        fig.savefig(f"mass_flow_run{ep + 1}.pdf")
        plt.show()



        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.step(time_step, sp_plot, where="post", label="Concentration setpoint")
        for i in range(4):
            ax.step(time_step, conc_plot[i], where="post", label=f"Zone {i+1}")
        ax.grid()
        ax.legend()
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(r"Concentration [kg/m$^3$]")
        ax.set_title(f"Concentrations left zones and Setpoint test run {ep+1}")
        fig.tight_layout()
        fig.savefig(f"concentrationLeft{ep + 1}.pdf")
        plt.show()

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.step(time_step, sp_plot, where="post", label="Concentration setpoint")
        for i in range(4, 8):
            ax.step(time_step, conc_plot[i], where="post", label=f"Zone {i + 1}")
        ax.grid()
        ax.legend()
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(r"Concentration [kg/m$^3$]")
        ax.set_title(f"Concentrations right zones and Setpoint test run {ep+1}")
        fig.tight_layout()
        fig.savefig(f"concentrationRight{ep + 1}.pdf")
        plt.show()
