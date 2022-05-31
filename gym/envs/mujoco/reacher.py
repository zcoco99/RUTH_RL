import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        self.disc_obs_index = None# [6,7]
        self.lower_dim = False
        self.entity = None #[0, 0]
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)
        

    def step(self, a):
#        vec = self.get_body_com("fingertip")-self.get_body_com("target"+str(self.entity[0])+str(self.entity[1]))
#        reward_dist = - np.linalg.norm(vec)
#        reward_ctrl = - np.square(a).sum()
#        reward = reward_dist + reward_ctrl
        reward = 0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=None, reward_ctrl=None)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.init_qpos
        qpos[0:2] = self.np_random.uniform(low=-0.1, high=0.1, size=2) + qpos[0:2] #self.model.nq
        
        if self.entity is not None:
            curr_index = self.entity[0]*6+self.entity[1]*2 + 2
            while True:
                self.goal = self.np_random.uniform(low=-.26, high=.26, size=2)
                nplinalg = np.linalg.norm(self.goal)
                if nplinalg <= 0.26 and nplinalg >= 0.028:
                    break
            qpos[curr_index:curr_index+2] = self.goal
        
        qvel = self.init_qvel
        qvel[:2] = self.np_random.uniform(low=-.005, high=.005, size=2)
        qvel[2:] = 0
        self.set_state(qpos, qvel)
        
        return self._get_obs()

    def _get_obs(self):
#        theta = self.sim.data.qpos.flat[:2]
#        return [np.concatenate([
#            np.cos(theta),
#            np.sin(theta),
#            self.sim.data.qvel.flat[:2],
#            self.get_body_com("fingertip")[:2]
#        ]), None]
        return [self.get_body_com("fingertip"), None]
#    self.get_body_com("fingertip") - self.get_body_com("target"+str(self.entity[0])+str(self.entity[1]))
#    return np.concatenate([
#            np.cos(theta),
#            np.sin(theta),
#            self.sim.data.qpos.flat[2:],
#            self.sim.data.qvel.flat[:2],
#            self.get_body_com("fingertip")
#        ])
        
    
    
    
        
