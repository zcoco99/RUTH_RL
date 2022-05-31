import numpy as np
from gym import utils
from gym.envs.mujocost import mujoco_env


DEFAULT_CAMERA_CONFIG = {
    'distance': 4.0,
}


class AntsEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='ant.xml',
                 ctrl_cost_weight=0.5,
                 contact_cost_weight=5e-4,
                 healthy_reward=1.0,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.2, 1.0),
                 contact_force_range=(-0.00001, 0.00001),
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 rgb_rendering_tracking=True):
        utils.EzPickle.__init__(**locals())

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)
        
        
        self.disc_obs_index = [-2,-1]
        self.lower_dim = True
        
        self.goal = np.zeros(2)

        mujoco_env.MujocoEnv.__init__(self, xml_file, 10, rgb_rendering_tracking=rgb_rendering_tracking)

    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.sim.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces))
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z)
        return is_healthy

    @property
    def done(self):
        done = (not self.is_healthy
                if self._terminate_when_unhealthy
                else False)
        done = done or self.gooutbound()
        return done
#        return self.gooutbound()
    
    def gooutbound(self):
        d = (self.get_body_com("torso")[:2].copy()>10).sum()
        return d
        

    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        reward = xy_velocity[0]
        reward = 0
        
        # done = self.done
        done = False
        observation = self._get_obs()
        observation[-2:] = xy_velocity  
        info = {
            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],

            'x_velocity': x_velocity,
            'y_velocity': y_velocity,
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        
        observations = np.concatenate((position, velocity))
        
        return np.concatenate((observations, np.array([0, 0])), 0).copy() 

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv) 
#        self.goal = self.np_random.uniform(low=-10, high=10, size=2)
        # qpos[-2:] = self.goal
        # qvel[-2:] = 0
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
