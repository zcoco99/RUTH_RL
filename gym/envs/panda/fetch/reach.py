import os
from gym import utils
from gym.envs.panda import fetch_env


# Ensure we get the path separator correct on windows
MODEL_PATH = os.path.join('franka_panda', 'panda.urdf')


class FetchReachEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', version='DIRECT'):

        initial_qpos = [0.07, 0.083, -0.087, -1.685, 0.0096, 1.645, 0.766, None, None, 0.02, 0.02, None] # 0.57 043 0.01
        fetch_env.FetchEnv.__init__(
            self, MODEL_PATH, version, offset=[0,0,0], pandaNumDofs=7, n_substeps=5000, force_substeps=5*240., 
            camera=[1.5,-90,-15,[0,0,0]], initial_qpos=initial_qpos, 
            has_object=False, block_gripper=True, 
            target_range=[[0.3,0.15,-0.25], [0.8,0.65,0.25]], distance_threshold=0.01,
            reward_type=reward_type)
        utils.EzPickle.__init__(self)
