import gym
import os
import time
import numpy as np
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
from typing import Callable

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
def decay_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Decay learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return initial_value * np.exp(progress_remaining)

    return func



model_dir = "models/SAC"
model_dir2 = "models/TD3"


log_dir = "tmp/ur5/"
# log_dir = f"tmp/log/SAC-{int(time.time())}"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(model_dir2, exist_ok=True)


env = gym.make('kelin-v0',version="DIRECT")
env = Monitor(env, log_dir)
model = SAC("MlpPolicy",env, verbose=1, tensorboard_log=log_dir,  ent_coef='auto', batch_size=1024, learning_starts=1000) #learning_rate=decay_schedule(0.001)
model2 = TD3("MlpPolicy",env, verbose=1, tensorboard_log=log_dir, batch_size=1024)
callback = SaveOnBestTrainingRewardCallback(check_freq=5e3, log_dir=log_dir)



TIMESTEPS = 3e5
for i in range(2):
    # model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.learn(total_timesteps=TIMESTEPS, callback = callback, eval_freq=1000, eval_log_path=log_dir)
    model.save(f"{model_dir}/{int(time.time())}")
    # model2.learn(total_timesteps=TIMESTEPS, callback = callback, eval_freq=1000, eval_log_path=log_dir)
    # model2.save(f"{model_dir2}/{int(time.time())}")

