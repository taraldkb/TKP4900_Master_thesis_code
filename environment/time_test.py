import torch
import json
from datetime import datetime
from environment.water_injection_env import WaterInjectionEnv
from environment.cfd_interface import run_cfd_step
from rl.network import PolicyNet
import csv
import numpy as np


# read config
with open("configs/RL_config.json", "r") as f:
    CONFIG = json.load(f)


def test_time():

    episode_log = "logs/time_test.csv"

    episode_total = []
    sim_time = []

    for i in range(6):
        case = i+1
        policy_path = f"Save/Case{case}.pt"
        env = WaterInjectionEnv(run_cfd_step, case)
        policy = PolicyNet(CONFIG["state_dim"], CONFIG["action_dim"], CONFIG["hidden_size"])
        policy.load_state_dict(torch.load(policy_path))
        policy.eval()
        wind_profile = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        sp_profile = [20.0, 25.0, 25.0, 25.0, 30.0, 30.0, 10.0, 10.0, 10.0, 10.0]
        counter = 0
        time_keeper = []
        episode_start = datetime.now()
        state = env.testing_reset()

        while not done:
            sim_start = datetime.now()
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32)
                mean, _ = policy(state_tensor)
                action = mean.numpy()

            state, reward, done, _ = env.test_step(action, wind_profile[counter], sp_profile[counter])
            td = datetime.now()-sim_start
            time_keeper.append(td.total_seconds()/60)

        ep_time = datetime.now()-episode_start
        sim_time.append(time_keeper)
        episode_total.append(ep_time.total_seconds()/60)

    episode_avg = []
    for i in range(len(sim_time)):
        episode_avg.append(np.mean(sim_time[i]))

    with open(episode_log, mode="w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["case", "episode", "avg"])

        for i in range(len(episode_total)):
            csv_writer.writerow([i+1, episode_total[i], episode_avg[i]])

