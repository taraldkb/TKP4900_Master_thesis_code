import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
from environment.water_injection_env import WaterInjectionEnv
from environment.cfd_interface import run_cfd_step

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







