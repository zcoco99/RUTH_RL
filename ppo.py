# Stable-baseline3 PPO
import gym
import numpy as np
import pybullet
import os

from stable_baselines3 import PPO, SAC
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}.zip")
                  self.model.save(self.save_path)

        return True

models_dir = "models/PPO"
models_path = f"{models_dir}/170000.zip"

logdir = "logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)


# Create and wrap the environment
env = gym.make('kelin-v0',version="DIRECT")
model = PPO("MlpPolicy", env, verbose = 1)

callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=logdir)


TIMESTEPS = 10000
for i in range(1):
	# Train the agent 
	model.learn(total_timesteps=1)
	model.save(f"{models_dir}/{TIMESTEPS*i}")





mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")