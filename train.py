import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
import os


environment_name = "CarRacing-v2"
env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=16)

action_space = env.action_space
observation_space = env.observation_space
print("Action space:", action_space)
print("Observation space:", observation_space)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

log_path = os.path.join("training", "logs")
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_path, device=device)
model.learn(total_timesteps=1_000_000)
model.save("training/ppo_car_racing")

evaluate_policy(model, env, n_eval_episodes=10, render=True)
env.close()
