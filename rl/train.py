import csv
import os
import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
import json
from environment.water_injection_env import WaterInjectionEnv
from environment.cfd_interface import run_cfd_step
from datetime import date
from torch.optim.lr_scheduler import StepLR
from rl.network import PolicyNet, ValueNet
from utils.training_functions import compute_returns, normalize

# Load config
with open("configs/RL_config.json", "r") as f:
    CONFIG = json.load(f)


# create Train agent
def train_agent(case, log_name=None):

    if log_name is None:
        log_name = str(date.today())
    log_name = log_name + ".csv"
    log_file = os.path.join(CONFIG['log_dir'], log_name)

    env = WaterInjectionEnv(run_cfd_step, case)
    policy = PolicyNet(
        CONFIG["state_dim"],
        CONFIG["action_dim"],
        CONFIG["hidden_size"])

    value = ValueNet(
        CONFIG["state_dim"],
        CONFIG["hidden_size"])

    optimizer_policy = optim.Adam(
        policy.parameters(), lr=CONFIG["lr_policy"])
    optimizer_value = optim.Adam(
        value.parameters(), lr=CONFIG["lr_value"])

    # create learning rate schedulers
    scheduler_policy = StepLR(
        optimizer_policy, step_size=75, gamma=0.9)
    scheduler_value = StepLR(
        optimizer_value, step_size=75, gamma=0.9)

    best_reward = -float("inf")

    with open(log_file, mode="w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            ["episode", "reward", "policy_loss", "value_loss"])

        for episode in range(CONFIG["epochs"]):
            state = env.reset()
            log_probs, rewards, values = [], [], []
            done = False
            total_reward = 0

            while not done:
                state_tensor = torch.tensor(
                    state, dtype=torch.float32)
                mean, std = policy(state_tensor)
                dist = Normal(mean, std)
                action = dist.sample()
                if CONFIG["clip_actions"]:
                    action = action.clamp(0, 1)

                log_prob = dist.log_prob(action).sum()
                value_est = value(state_tensor)

                next_state, reward, done, _ = env.step(
                    action.detach().numpy())

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

            csv_writer.writerow(
                [episode, total_reward, policy_loss.item(), value_loss.item()])
            csvfile.flush()

            # step scheduler
            scheduler_policy.step()
            scheduler_value.step()

            if total_reward > best_reward:
                best_reward = total_reward
                torch.save(policy.state_dict(), CONFIG["save_path"])

    print("Training complete. Best reward:", best_reward)


def continue_train_agent(policy_path, log_path, case, episodes=100):

    env = WaterInjectionEnv(run_cfd_step, case)
    policy = PolicyNet(
        CONFIG["state_dim"],
        CONFIG["action_dim"],
        CONFIG["hidden_size"])

    policy.load_state_dict(torch.load(policy_path))

    value = ValueNet(
        CONFIG["state_dim"],
        CONFIG["hidden_size"])

    optimizer_policy = optim.Adam(
        policy.parameters(), lr=CONFIG["lr_policy"])
    optimizer_value = optim.Adam(
        value.parameters(), lr=CONFIG["lr_value"])

    # create learning rate schedulers
    scheduler_policy = StepLR(
        optimizer_policy, step_size=75, gamma=0.9)
    scheduler_value = StepLR(
        optimizer_value, step_size=75, gamma=0.9)

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
            # create header if new file
            csv_writer.writerow(
                ["episode", "reward", "policy_loss", "value_loss"])

        for episode in range(episodes):
            episode_number = last_episode + 1 + episode
            state = env.reset()
            log_probs, rewards, values = [], [], []
            done = False
            total_reward = 0

            while not done:
                state_tensor = torch.tensor(
                    state, dtype=torch.float32)

                mean, std = policy(state_tensor)
                dist = Normal(mean, std)
                action = dist.sample()
                if CONFIG["clip_actions"]:
                    action = action.clamp(0, 1)

                log_prob = dist.log_prob(action).sum()
                value_est = value(state_tensor)

                next_state, reward, done, _ = env.step(
                    action.detach().numpy())

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

            csv_writer.writerow(
                [episode_number, total_reward, policy_loss.item(), value_loss.item()])

            csvfile.flush()
            # step scheduler
            scheduler_policy.step()
            scheduler_value.step()

            if total_reward > best_reward:
                best_reward = total_reward
                torch.save(policy.state_dict(), policy_path)

    print("Training complete. Best reward:", best_reward)
