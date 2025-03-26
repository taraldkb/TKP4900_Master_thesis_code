import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.water_injection_env import WaterInjectionEnv
from environment.cfd_interface import run_cfd_step


def train_ppo(total_timesteps=1000, save_path="Save/ppo_water_agent", tensorboard_log = "logs/ppo/"):

    env = DummyVecEnv([lambda: WaterInjectionEnv(run_cfd_step_fn=run_cfd_step)])
    check_env(env.envs[0])

    # create PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=tensorboard_log
    )

    # train the agent
    model.learn(total_timesteps=total_timesteps)

    # Save model
    model.save(save_path)


def test_ppo(model_path="Save/ppo_water_agent.zip", episodes=5):

    env = WaterInjectionEnv(run_cfd_step_fn=run_cfd_step)
    model = PPO.load(model_path)

    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        print(f"Episode {ep + 1} reward: {total_reward:.2f}")

