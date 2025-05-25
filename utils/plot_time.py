import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np


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

col1 = '#1f77b4'
col2 = '#ff7f0e'

fig_width = 6.27
fig_height = fig_width * 0.618


def plot_time(time_logs):

    time_data = pd.read_csv(time_logs)
    cases = ["Case 1", "Case 2", "Case 3", "Case 4", "Case 5", "Case 6"]

    episode_total = time_data["episode"]
    simulation_avg = time_data["avg"]

    x = np.arange(len(cases))
    width = 0.25
    fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))

    ax2 = ax1.twinx()

    bar1 = ax1.bar(x-width/2, episode_total, width, label="Average episode total time", color=col1)
    bar2 = ax2.bar(x + width / 2, simulation_avg, width, label="Average simulation step time", color=col2)

    ax1.set_xticks(x)
    ax1.set_xticklabels(cases)

    ax1.set_ylabel("Episode time [m]")
    ax2.set_ylabel("Step time [m]")
    ax1.set_title("Average episode and simulation step time")

    bars = [bar1, bar2]
    labels = [bar.get_label() for bar in bars]
    ax1.legend(bars, labels, loc='upper center', ncol=1)

    plt.tight_layout()
    fig.savefig("time.pdf")
    plt.show()



