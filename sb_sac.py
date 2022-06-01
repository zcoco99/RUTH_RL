# Stable-baseline3 SAC
import gym
import numpy as np
import pybullet
import os

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import results_plotter


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


# Create and wrap the environment
log_dir = "./tmp/log/"
os.makedirs(log_dir, exist_ok=True)

env = gym.make('kelin-v0',version="DIRECT")
env = Monitor(env, log_dir)

# Create the callback: check every 1000 steps
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

# Evaluate the model every 1000 steps on 5 test episodes and save the evaluation to the logs folder
model = SAC("MlpPolicy", env, verbose = 1, learning_rate=1e-3, create_eval_env=True)
model.learn(total_timesteps=int(5e4), callback=callback, eval_freq=1000, n_eval_episodes=5, eval_log_path=log_dir)


#Save policy only
policy = model.policy
policy.save("sac_policy.pkl")
env = model.get_env()


results_plotter.plot_results([log_dir], 1e5, results_plotter.X_TIMESTEPS, "SAC output")


#Evaluate policy
# mean_reward, std_reward = evaluate_policy(policy, env, n_eval_episodes=10, deterministic=True)
# print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

# Evaluate the loaded policy
# saved_policy = MlpPolicy.load("sac_policy.pkl")
# mean_reward, std_reward = evaluate_policy(saved_policy, env, n_eval_episodes=10, deterministic=True)
# print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")