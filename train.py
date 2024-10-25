import torch
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
print("Device:", device)

environment_name = "CarRacing-v2"
# Training environment setup
env = gym.make(environment_name)
env = Monitor(env)  # Wrap in Monitor to track episode info
env = DummyVecEnv(
    [lambda: env]
)  # Wrap in DummyVecEnv (VecTransposeImage will be done automatically)

# Evaluation environment setup
eval_env = gym.make(environment_name)
eval_env = Monitor(eval_env)  # Wrap in Monitor for evaluation environment
eval_env = DummyVecEnv([lambda: eval_env])  # Wrap in DummyVecEnv
eval_env = VecTransposeImage(
    eval_env
)  # Ensure correct image format for CNNs for evaluation

log_path = os.path.join("training", "logs")
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log=log_path,
    device=device,
    learning_rate=1e-4,
)

# eval_env = gym.make(environment_name)
best_model_save_path = "./best_model/"
os.makedirs(best_model_save_path, exist_ok=True)

stop_callback = StopTrainingOnRewardThreshold(
    reward_threshold=1175,
    verbose=1,
)

# Custom evaluation callback to handle tolerance in step counting
eval_callback = EvalCallback(
    eval_env=eval_env,
    best_model_save_path=best_model_save_path,
    log_path="./logs/",
    eval_freq=100000,
    n_eval_episodes=5,
    deterministic=True,
    render=False,
    callback_on_new_best=stop_callback,
    verbose=1,
)

model.learn(total_timesteps=5_000_000, callback=eval_callback, progress_bar=True)
model.save("training/ppo_car_racing")

evaluate_policy(model, eval_env, n_eval_episodes=10, render=False)
env.close() 
