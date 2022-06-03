import gym
import numpy as np
from stable_baselines3 import SAC 


model_dir = "models/SAC/"
model_path = f"{model_dir}/0.zip "
env = gym.make('kelin-v0',version="GUI")
model = SAC.load(model_path, env=env)

episodes = 500

for ep in range (episodes):
    obs = env.reset()
    done = False
    while not done:
        action, state = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        print(reward)
