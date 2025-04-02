import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from environment.water_injection_env import WaterInjectionEnv
from environment.cfd_interface import run_cfd_step


# define policy network

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Sigmoid()
        )

    def forward(self, state):
        return self.net(state)


# define the compute discounted return function
def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)


# create training agent and loop
def train_agent(epochs=100, gamma=0.99, lr=1e-3, save_path="Save/policy.pt", log_dir="logs/reinforce"):
    env = WaterInjectionEnv(run_cfd_step)
    policy = PolicyNet(state_dim=10, action_dim=3)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    writer = SummaryWriter(log_dir=log_dir)

    for episode in range(epochs):
        state = env.reset()
        log_probs = []
        rewards = []

        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action = policy(state_tensor)
            dist = torch.distributions.Uniform(0, 1)
            sampled_action = action.detach().numpy()

            # Step det cfd model
            next_state, reward, done, _ = env.step(sampled_action)
            rewards.append(reward)

            # calculate and store log prob
            log_prob = torch.sum(torch.log(action + 1e-8))
            log_probs.append(log_prob)

            state = next_state

        # compute return and loss
        returns = compute_returns(rewards, gamma)
        log_probs = torch.stack(log_probs)
        loss = -torch.sum(log_probs*returns)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward = sum(rewards)
        print(f"Episode {episode + 1}: total reward = {total_reward:.2f}, loss = {loss.item():.2f}")

        # log to tensor board
        writer.add_scalar("reward", total_reward, episode)
        writer.add_scalar("loss", loss.item, episode)

    writer.close()
    torch.save(policy.state_dict(), save_path)


# Create testing loop
def test_agent(policy_path="policy.pt", episodes=5):
    env = WaterInjectionEnv(run_cfd_step)
    policy = PolicyNet(state_dim=9, action_dim=3)
    policy.load_state_dict(torch.load(policy_path))
    policy.eval()

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                action = policy(torch.tensor(state, dtype=torch.float32)).numpy()
            state, reward, done, _ = env.step(action)
            total_reward += reward
        print(f"[Test] Episode {ep+1}: reward = {total_reward:.2f}")



