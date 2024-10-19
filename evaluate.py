import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# Set the device to MPS if available, otherwise use CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

# Create a new environment for evaluation
environment_name = "CarRacing-v2"
eval_env = gym.make(environment_name, render_mode="human")

# Wrap with Monitor for evaluation
eval_env = Monitor(eval_env)

# DummyVecEnv and VecFrameStack to match training setup
eval_env = DummyVecEnv([lambda: eval_env])
eval_env = VecFrameStack(eval_env, n_stack=16)

# Load the model onto the correct device
model = PPO.load("training/ppo_car_racing", device=device)

# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, render=True)

print(f"Mean reward: {mean_reward}, Standard deviation: {std_reward}")

# Close the evaluation environment
eval_env.close()
