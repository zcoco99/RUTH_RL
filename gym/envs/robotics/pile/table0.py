import os
from gym import utils
from gym.envs.robotics import pile_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('pile', 'table012.xml')


class PileTable0Env(pile_env.PileEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
                'robot':{
                        'robot0:slide0': 0.405,
                        'robot0:slide1': 0.48,
                        'robot0:slide2': 0.0,
                        },
                'object':{
                        'object:joint:center': [1.3, 0.54, -0.03, 1., 0., 0., 0.],
                        'object:joint:size': [0.025, 0.025, 0.025],
                        'object:joint:range': [0.0, 0.0, 0.0],
                        },
        }
        pile_env.PileEnv.__init__(
            self, MODEL_XML_PATH, object_num=0, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, obs_index=None,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
        
        self.disc_obs_index = [0,1]
        self.lower_dim = True
    
    def _is_done(self, obs, rew=False):
        
        if self.object_num[0] == 1:
            x, y, z = obs[6], obs[7], obs[8]
            xd = x <= 1.06 or x >= 1.54
            yd = y <= 0.34 or y >= 0.74
            zd = z< 0.42
            
            mx = abs(obs[0] - self.object_range['object:joint:center'][0]) >=0.005
            my = abs(obs[1] - self.object_range['object:joint:center'][1]) >=0.005
            mz = abs(obs[2] - self.object_range['object:joint:center'][2]) >=0.005
            d = xd or yd or zd or mx or my or mz
#            if not rew:
#                d = zd or mx or my or mz
        else:
            x, y, z = obs[0], obs[1], obs[2]
            xd = x <= 1.06 or x >= 1.54
            yd = y <= 0.34 or y >= 0.74
            zd = z< 0.42
            d = xd or yd or zd
#            if not rew:
#                d = zd
        
        return d
    
    def compute_reward(self, obs):
        # Compute distance between goal and the achieved goal.
        z = obs[2]
        
#        print(z)#0.42
        return (0.5 - z)*12.5# + self._is_done(obs, rew=True)
#        if self.reward_type == 'sparse':
#            if z <= 0.45:
#                return 0
#            else:
#                return -1 
#        else:
#            return 0.425 - z 