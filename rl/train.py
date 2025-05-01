import csv
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import json
from environment.water_injection_env import WaterInjectionEnv
from environment.cfd_interface import run_cfd_step
from utils.map_value_function import *
from utils.read_report_function import *
from datetime import date
import matplotlib.pyplot as plt

# Load config
with open("configs/RL_config.json", "r") as f:
    CONFIG = json.load(f)


# create Policy Network
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, state):
        x = self.net(state)
        mean = torch.sigmoid(self.mean_head(x))
        std = torch.exp(self.log_std)
        return mean, std


# create value net for baseline
class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden=64):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, state):
        return self.value(state).squeeze(-1)


# create utility functions
def compute_returns(rewards, gamma):
    returns = []
    G = 0

    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    return torch.tensor(returns, dtype=torch.float32)


def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-8)


# create Train agent
def train_agent(log_name=None):

    if log_name is None:
        log_name = str(date.today())
    log_name = log_name + ".csv"
    log_file = os.path.join(CONFIG['log_dir'], log_name)

    env = WaterInjectionEnv(run_cfd_step)
    policy = PolicyNet(CONFIG["state_dim"], CONFIG["action_dim"], CONFIG["hidden_size"])
    value = ValueNet(CONFIG["state_dim"], CONFIG["hidden_size"])

    optimizer_policy = optim.Adam(policy.parameters(), lr=CONFIG["lr_policy"])
    optimizer_value = optim.Adam(value.parameters(), lr=CONFIG["lr_value"])

    best_reward = -float("inf")

    with open(log_file, mode="w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["episode", "reward", "policy_loss","value_loss"])

        for episode in range(CONFIG["epochs"]):
            state = env.reset()
            log_probs, rewards, values = [], [], []
            done = False
            total_reward = 0

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32)
                mean, std = policy(state_tensor)
                dist = Normal(mean, std)
                action = dist.sample()
                if CONFIG["clip_actions"]:
                    action = action.clamp(0, 1)

                log_prob = dist.log_prob(action).sum()
                value_est = value(state_tensor)

                next_state, reward, done, _ = env.step(action.detach().numpy())

                log_probs.append(log_prob)
                rewards.append(reward)
                values.append(value_est)
                state = next_state
                total_reward += reward

            returns = compute_returns(rewards, CONFIG["gamma"])
            log_probs = torch.stack(log_probs)
            values = torch.stack(values)
            advantages = normalize(returns - values.detach())

            policy_loss = -(log_probs * advantages).sum()
            optimizer_policy.zero_grad()
            policy_loss.backward()
            optimizer_policy.step()

            value_loss = nn.functional.mse_loss(values, returns)
            optimizer_value.zero_grad()
            value_loss.backward()
            optimizer_value.step()

            csv_writer.writerow([episode, total_reward, policy_loss.item(), value_loss.item()])
            csvfile.flush()

            if total_reward > best_reward:
                best_reward = total_reward
                torch.save(policy.state_dict(), CONFIG["save_path"])

    print("Training complete. Best reward:", best_reward)


def continue_train_agent(policy_path, log_path, episodes=100):

    env = WaterInjectionEnv(run_cfd_step)
    policy = PolicyNet(CONFIG["state_dim"], CONFIG["action_dim"], CONFIG["hidden_size"])
    policy.load_state_dict(torch.load(policy_path))
    value = ValueNet(CONFIG["state_dim"], CONFIG["hidden_size"])

    optimizer_policy = optim.Adam(policy.parameters(), lr=CONFIG["lr_policy"])
    optimizer_value = optim.Adam(value.parameters(), lr=CONFIG["lr_value"])

    # find last recorded episode and current best reward
    last_episode = -1
    best_reward = -float("inf")

    if os.path.exists(log_path):  # check file exists
        with open(log_path, mode="r") as csvfile:
            reader = csv.reader(csvfile)
            lines = list(reader)

            if len(lines) > 1:  # check for existing lines in log
                last_episode = int(lines[-1][0])
                rewards = [float(row[1]) for row in lines[1:] if len(row) > 1]
                if rewards:
                    best_reward = max(rewards)
            else:
                last_episode = -1
                best_reward = -float("inf")

    else:  # make file if it does not exist
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        last_episode = -1

    with open(log_path, mode="a", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if last_episode == -1:
            csv_writer.writerow(["episode", "reward", "policy_loss", "value_loss"])  # create header if new file

        for episode in range(episodes):
            episode_number = last_episode + 1 + episode
            state = env.reset()
            log_probs, rewards, values = [], [], []
            done = False
            total_reward = 0

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32)
                mean, std = policy(state_tensor)
                dist = Normal(mean, std)
                action = dist.sample()
                if CONFIG["clip_actions"]:
                    action = action.clamp(0, 1)

                log_prob = dist.log_prob(action).sum()
                value_est = value(state_tensor)

                next_state, reward, done, _ = env.step(action.detach().numpy())

                log_probs.append(log_prob)
                rewards.append(reward)
                values.append(value_est)
                state = next_state
                total_reward += reward

            returns = compute_returns(rewards, CONFIG["gamma"])
            log_probs = torch.stack(log_probs)
            values = torch.stack(values)
            advantages = normalize(returns - values.detach())

            policy_loss = -(log_probs * advantages).sum()
            optimizer_policy.zero_grad()
            policy_loss.backward()
            optimizer_policy.step()

            value_loss = nn.functional.mse_loss(values, returns)
            optimizer_value.zero_grad()
            value_loss.backward()
            optimizer_value.step()

            csv_writer.writerow([episode_number, total_reward, policy_loss.item(), value_loss.item()])
            csvfile.flush()

            if total_reward > best_reward:
                best_reward = total_reward
                torch.save(policy.state_dict(), policy_path)

    print("Training complete. Best reward:", best_reward)


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
        injection1 = [map_value(0.25, 0, 10)]
        injection2 = [map_value(0.25, 0, 10)]
        mass = [map_value(0.5, 0, 100)]
        conc_plot = [[] for _ in range(8)]
        wind_plot = [state[-2]]
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
            wind_plot.append(state[-2])
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
        plt.xlabel("Time step [s]")
        plt.ylabel("Velocity [m/s]")
        plt.title(f"Injection velocity test run {ep+1}")
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.step(time_step, mass, where="post")
        plt.grid()
        plt.xlabel("Time step [s]")
        plt.ylabel("Mass flow [kg/s check this !!!!!]")
        plt.title(f"Injection mass flow test run {ep+1}")
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.step(time_step, wind_plot, where="post")
        plt.grid()
        plt.xlabel("Time step [s]")
        plt.ylabel("Velocity [m/s]")
        plt.title(f"Wind velocity test run {ep+1}")
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.step(time_step, sp_plot, where="post", label="Concentration setpoint")
        for i in range(4):
            plt.step(time_step, conc_plot[i], where="post", label=f"Zone {i+1}")
        plt.grid()
        plt.legend()
        plt.xlabel("Time step [s]")
        plt.ylabel("Concentration [INSERT Units!!!!]")
        plt.title(f"Concentrations left zones and Setpoint test run {ep+1}")
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.step(time_step, sp_plot, where="post", label="Concentration setpoint")
        for i in range(4, 8):
            plt.step(time_step, conc_plot[i], where="post", label=f"Zone {i + 1}")
        plt.grid()
        plt.legend()
        plt.xlabel("Time step [s]")
        plt.ylabel("Concentration [INSERT Units!!!!]")
        plt.title(f"Concentrations right zones and Setpoint test run {ep+1}")
        plt.show()















