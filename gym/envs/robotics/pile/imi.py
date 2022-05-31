import os
from gym import utils
from gym.envs.robotics import pile_env
import numpy as np

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('pile', 'imi.xml')


class PileImiEnv(pile_env.PileEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
                'robot':{
                        'robot0:slide0': 0.405,
                        'robot0:slide1': 0.48,
                        'robot0:slide2': 0.0,
                        },
                'object':{
                        'object:joint:center': [1.3, 0.75, 0.425, 1., 0., 0., 0.],
                        'object:joint:size': [0.025, 0.025, 0.025],
                        'object:joint:range': [0.005, 0.005],
                        },
        }
        pile_env.PileEnv.__init__(
            self, MODEL_XML_PATH, object_num=0, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, obs_index=[0,1],
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
        self.disc_obs_index = None
        self.lower_dim = False
        
    def _is_done(self, obs):
        return 0
    
    def compute_reward(self, obs):
#        z = 0.425 -obs[0][-3] + np.abs(obs[0][-5:-3]-obs[0][:2]).sum()
        
        r = np.sqrt(((obs[0][-3:]-[1.34, 0.75, 0.535])**2).sum())
        return  r