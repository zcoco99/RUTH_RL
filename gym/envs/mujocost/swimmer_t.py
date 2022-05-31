import numpy as np
from gym.envs.mujocost import mujoco_env
from gym import utils


DEFAULT_CAMERA_CONFIG = {}


class SwimmertEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='swimmer.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=1e-4,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 rgb_rendering_tracking=True):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)
        
        self.disc_obs_index = [-2,-1]
        self.lower_dim = True 

        mujoco_env.MujocoEnv.__init__(self, xml_file, 10, rgb_rendering_tracking=rgb_rendering_tracking)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        action[0] = 0 
        xy_position_before = self.sim.data.qpos[0:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.sim.data.qpos[0:2].copy()
        xy_velocity = (xy_position_after - xy_position_before) / self.dt 

        observation = self._get_obs()
        observation[-2:] = xy_velocity 
        # Avector = (xy_position_after + xy_position_before)/2
        # Bvector = xy_position_after - xy_position_before
        # reward = np.sum(Avector * Bvector)/ \
        #     ( np.linalg.norm(Avector)*np.linalg.norm(Bvector) )
        # reward = (reward - 1)/1.0 
        # reward = np.linalg.norm(xy_velocity)
        reward = 0
        done = False
        info = {
            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        # velocity = self.sim.data.qvel.flat.copy()

        # if self._exclude_current_positions_from_observation:
        #     position = position[2:]
        # return position
        return np.concatenate((position[2:], np.array([0, 0])), 0).copy() 

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
    
    def resetmiddle(self):
        return self._get_obs() 

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
