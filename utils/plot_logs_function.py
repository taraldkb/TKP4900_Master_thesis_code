import matplotlib.pyplot as plt
import pandas as pd


def plot_logs(log_path):

    logs = pd.read_csv(log_path)

    # unpack variables
    episodes = logs["episode"]
    rewards = logs["reward"]
    policy_loss = logs["policy_loss"]
    value_loss = logs["value_loss"]

    # plot rewards
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, label="Reward")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Total reward per episode")
    plt.legend()
    plt.grid()
    plt.show()

    # plot rewards
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, policy_loss, label="Policy Loss")
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.title("Policy loss per episode")
    plt.legend()
    plt.grid()
    plt.show()

    # plot rewards
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, value_loss, label="Value Loss")
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.title("Value loss per episode")
    plt.legend()
    plt.grid()
    plt.show()


