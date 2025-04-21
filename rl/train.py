import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
from environment.water_injection_env import WaterInjectionEnv
from environment.cfd_interface import run_cfd_step
from utils.map_value_function import *
from utils.read_report_function import *
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
def train_agent():
    writer = SummaryWriter(log_dir=CONFIG["log_dir"])

    env = WaterInjectionEnv(run_cfd_step)
    policy = PolicyNet(CONFIG["state_dim"], CONFIG["action_dim"], CONFIG["hidden_size"])
    value = ValueNet(CONFIG["state_dim"], CONFIG["hidden_size"])

    optimizer_policy = optim.Adam(policy.parameters(), lr=CONFIG["lr_policy"])
    optimizer_value = optim.Adam(value.parameters(), lr=CONFIG["lr_value"])

    best_reward = -float("inf")

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

        writer.add_scalar("reward", total_reward, episode)
        writer.add_scalar("loss/policy", policy_loss.item(), episode)
        writer.add_scalar("loss/value", value_loss.item(), episode)

        print(
            f"Ep {episode + 1}: Reward={total_reward:.2f}, PolicyLoss={policy_loss.item():.3f}, "
            f"ValueLoss={value_loss.item():.3f}")

        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(policy.state_dict(), CONFIG["save_path"])
    writer.close()
    print("Training complete. Best reward:", best_reward)


# create tester
def test_agent(policy_path=CONFIG["save_path"]):
    env = WaterInjectionEnv(run_cfd_step)
    policy = PolicyNet(CONFIG["state_dim"], CONFIG["action_dim"], CONFIG["hidden_size"])
    policy.load_state_dict(torch.load(policy_path))
    policy.eval()

    wind_profiles_lib = [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                         [0.5, 0.95, 0.95, 0.95, 0.95, 0.2, 0.2, 0.2, 0.2, 0.2],
                         [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 0.8]]
    sp_profiles_lib = [[20.0, 25.0, 25.0, 25.0, 30.0, 30.0, 10.0, 10.0, 10.0, 10.0],
                       [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
                       [20.0, 25.0, 30.0, 30.0,  30.0, 30.0, 30.0, 30.0, 30.0, 30.0]]

    for ep in range(3):
        wind_profile = wind_profiles_lib[ep]
        sp_profile = sp_profiles_lib[ep]
        injection1 = []
        injection2 = []
        mass = []
        state = env.testing_reset()
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
                mass.append(map_value(action[2], 0, 100))

            state, reward, done, _ = env.step(action)
            total_reward += reward

            # save reward and state for plotting
            total_rewards.append(total_reward)
            rewards.append(reward)
            counter += 1
        plot_conc("concentration.out", ep)




        print(f"[Test] Episode {ep + 1}: reward = {total_reward:.2f}")








