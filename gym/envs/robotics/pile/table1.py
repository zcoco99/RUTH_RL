import os
from gym import utils
from gym.envs.robotics import pile_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('pile', 'table.xml')


class PileTableEnv(pile_env.PileEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
                'robot':{
                        'robot0:slide0': 0.405,
                        'robot0:slide1': 0.48,
                        'robot0:slide2': 0.0,
                        },
                'object':{
                        'object:joint:center': [1.3, 0.54, 0.425, 1., 0., 0., 0.],
                        'object:joint:size': [0.025, 0.025, 0.025],
                        'object:joint:range': [0.25, 0.21],
                        },
        }
        pile_env.PileEnv.__init__(
            self, MODEL_XML_PATH, object_num=3, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, 
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
