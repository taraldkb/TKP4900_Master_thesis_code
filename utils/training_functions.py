import torch


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
