import os
from gym import utils
from gym.envs.robotics import pile_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('pile', 'group.xml')


class PileGroupEnv(pile_env.PileEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.63, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.35, 0.73, 0.4, 1., 0., 0., 0.],
            'object2:joint': [1.40, 0.50, 0.4, 1., 0., 0., 0.],
        }
        pile_env.PileEnv.__init__(
            self, MODEL_XML_PATH, object_num=3, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
