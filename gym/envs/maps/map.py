
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random


class MapEnv(gym.Env):
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.size = 1
        
        self.action_space = spaces.Box(np.array([-1.0,]*2), np.array([1.0,]*2))
        self.observation_space = spaces.Box(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        
        self.state = np.array([0.25, 0.25]) 
        self.max_delta_action = 0.05 
        
        self.disc_obs_index = None
        self.lower_dim = False 
        self.doneNum = 0  
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # action = np.clip(action, -1, 1) 
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        action = action * self.max_delta_action 
        for action in [action, action/2, action/4]:
        # for action in [action, action/2, [0.0,action[1]/2], [action[0]/2,0.0]]:
            state = self.state + action 
            if state[0]>0 and state[0]<1 and state[1]>0 and state[1]<1:
                self.state = state 
                break 
            self.doneNum += 1 
        
        r = - np.linalg.norm(self.state - [0.75, 0.25])
        d = (self.doneNum > 10)  
        d = False 
        
        return self.state, r, d, {}  

    def reset(self):
        self.state = np.array([0.25, 0.25]) 
        # self.state = np.random.rand(2) 
        self.doneNum = 0 
        return self.state
    
    def resetmiddle(self): 
        
        # if random.randint(0,1) == 0: # map a
        #     self.state = np.random.rand(2) * [0.20, 0.60] + np.array([0.20, 0.20]) 
        # else:
        #     self.state = np.random.rand(2) * [0.30, 0.35] + np.array([0.40, 0.51])
        
        # if random.randint(0,1) == 0: # map b
        #     self.state = np.random.rand(2) * [0.20, 0.7] + np.array([0.20, 0.20]) 
        # else:
        #     self.state = np.random.rand(2) * [0.30, 0.25] + np.array([0.40, 0.71])
            
        # randomindex = random.randint(0,2) # map b
        # if randomindex == 0:
        #     self.state = np.random.rand(2) * [0.20, 0.70] + np.array([0.20, 0.20]) 
        # if randomindex == 1:
        #     self.state = np.random.rand(2) * [0.30, 0.25] + np.array([0.40, 0.71])
        # if randomindex == 2:
        #     self.state = np.random.rand(2) * [0.30, 0.40] + np.array([0.55, 0.40])
        
        # if random.randint(0,1) == 0: # map c 
        #     self.state = np.random.rand(2) * [0.20, 0.5] + np.array([0.05, 0.20]) 
        # else:
        #     self.state = np.random.rand(2) * [0.70, 0.25] + np.array([0.10, 0.51])
        self.state = np.random.rand(2) 
        
        self.doneNum = 0 
        return self.state

#    def render(self, mode='human', **kwargs):
#        return self.env.render(mode, **kwargs)

    def render_frame(self, hw=15, imagesize=(300,300), camera_id=0, heatmap=None):
        hw = 100
        margin = 4
        screen_width = hw + margin*2
        screen_height = hw + margin*2
        if heatmap is None:
            frame = np.zeros([screen_width, screen_height, 3]) + 255
        else:
            frame = heatmap.copy() 
        xy = self.state * hw + margin 
        x = int(xy[0])
        y = int(xy[1])
        delta = 2 
        frame[x-delta:x+delta, y-delta:y+delta] = 0 
        frame[x-delta:x+delta, y-delta:y+delta, 0] = 0
        frame[x-delta:x+delta, y-delta:y+delta, 1] = 0
        frame[x-delta:x+delta, y-delta:y+delta, 2] = 255
        
        color=[140,180,210]
        frame[:margin,:, 0] = color[0]; frame[:margin,:,1] = color[1]; frame[:margin,:,2] = color[2];
        frame[-margin:,:, 0] = color[0]; frame[-margin:,:,1] = color[1]; frame[-margin:,:,2] = color[2];
        frame[:,:margin, 0] = color[0]; frame[:,:margin,1] = color[1]; frame[:,:margin,2] = color[2];
        frame[:,-margin::, 0] = color[0]; frame[:,-margin:,1] = color[1]; frame[:,-margin:,2] = color[2];
        # frame[:margin,:] = 128; frame[-margin:,:]=128;
        # frame[:,:margin] = 128; frame[:,-margin:]=128;
        return frame 

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None