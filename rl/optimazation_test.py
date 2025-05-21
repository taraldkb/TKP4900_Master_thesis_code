import torch
import json
from rl.network import PolicyNet
from utils.get_case_function import get_case


# read config
with open("configs/RL_config.json", "r") as f:
    CONFIG = json.load(f)


# create policy function for optimazation
def pi(state, H, N):

    # define wind an setpoint states
    wind = [0.5, 0.5, 0.75, 0.75]
    sp = [10, 20, 10, 20]

    # create saving array for water usage
    mass_flow = []

    # get correct policy file
    case = get_case(H, N)
    case_policy = "Save/"+case+".pt"

    policy = PolicyNet(CONFIG["state_dim"], CONFIG["action_dim"], CONFIG["hidden_size"])
    policy.load_state_dict(torch.load(case_policy))
    policy.eval()

    for i in range(len(wind)):
        state[-2] = wind[i]
        state[-1] = sp[1]

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            mean, _ = policy(state_tensor)
            action = mean.numpy()
            flow = N * action[-1]
            mass_flow.append(flow)

    return sum(mass_flow)




