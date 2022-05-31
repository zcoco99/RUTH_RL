import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


DEFAULT_CAMERA_CONFIG = {
    'distance': 4.0,
}


class PointEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='point.xml',
                 ctrl_cost_weight=0.5,
                 contact_cost_weight=5e-4,
                 healthy_reward=1.0,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.2, 1.0),
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 rgb_rendering_tracking=True):
        utils.EzPickle.__init__(**locals())
        
        self._reset_noise_scale = reset_noise_scale
        self.disc_obs_index = [0,1]
        self.lower_dim = True
        
        # self.goal = np.zeros(2)
        self._action_steps = 5

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5, rgb_rendering_tracking=rgb_rendering_tracking)

    
    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        # num_steps = np.random.randint( int(self._action_steps-2), self._action_steps)
        # for _ in range(num_steps):
        #     self.do_simulation(action, self.frame_skip)
        num_steps = self._action_steps
        for _ in range(num_steps):
            self.do_simulation(action, self.frame_skip)
        qpos = self.sim.data.qpos.copy()
        qvel = self.sim.data.qvel.copy()*0.0
        self.set_state(qpos, qvel)
        # for _ in range(num_steps):
            # self.do_simulation(-action, self.frame_skip)
            
        xy_position_after = self.get_body_com("torso")[:2].copy()

        reward = np.sqrt(((xy_position_after-[3.5, 3.5])**2).sum())/15
        
        done = False
        observation = self._get_obs()
        info = {
            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

#        if self._exclude_current_positions_from_observation:
#            position = position[2:]
        xy = self.get_body_com("torso")[:2].copy()
        observations = np.concatenate((xy/15, position/15, velocity))
#        observations = position

        return observations

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)
        
#        self.goal = self.np_random.uniform(low=-10, high=10, size=2)
        qpos[-2:] = np.zeros(2) 
        qvel[-2:] = 0
        
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation
    
    # def setgoal(self, goal):
    #     self.goal = goal

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
