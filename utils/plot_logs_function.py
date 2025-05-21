import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

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


def plot_logs(log_path):

    filename = log_path.split('/')[-1].split('.')[0]
    logs = pd.read_csv(log_path)

    # unpack variables
    episodes = logs["episode"]
    rewards = logs["reward"]
    policy_loss = logs["policy_loss"]
    value_loss = logs["value_loss"]

    # plot rewards
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.plot(episodes, rewards, label="Reward")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Reward")
    ax.set_title("Total reward per episode")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(filename+"_Rewards.pdf")
    plt.show()

    # plot policy loss
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.plot(episodes, policy_loss, label="Policy Loss")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Loss")
    ax.set_title("Policy loss per episode")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(filename + "_PolicyLoss.pdf")
    plt.show()

    # plot value loss
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.plot(episodes, value_loss, label="ValueLoss")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Loss")
    ax.set_title("Value loss per episode")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(filename + "_ValueLoss.pdf")
    plt.show()




