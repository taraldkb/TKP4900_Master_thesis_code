from environment.water_injection_env import WaterInjectionEnv
from environment.cfd_interface import run_cfd_step


# run initialization process to get a initial state for all test.
def get_state():
    env = WaterInjectionEnv(run_cfd_step, 1)
    state = env.testing_reset()
    return state

