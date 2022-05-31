
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random


class MapcEnv(gym.Env):
    
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
        self.barw = 0.03
        self.barh = 0.25
        
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
        for action in [action, action/3]:
        # for action in [action, action/2, [0.0,action[1]/2], [action[0]/2,0.0]]:
            state = self.state + action 
            inmap = (state[0]>0 and state[0]<1 and state[1]>0 and state[1]<1)
            inbarleft = (state[0]>0.5-self.barw and state[0]<0.5+self.barw and state[1]>0.5-self.barh and state[1]<0.5+self.barw)
            inbarup = (state[1]>0.5-self.barw and state[1]<0.5+self.barw and state[0]>0.5-self.barh and state[0]<0.5+self.barw)
            if inmap and (not inbarleft) and (not inbarup):
                self.state = state
                break 
            self.doneNum += 1
        
        r = - np.linalg.norm(self.state - [0.75, 0.75])
        d = (self.doneNum > 5)  
        d = False 
        
        return self.state, r, d, {}  

    def reset(self):
        self.state = np.array([0.25, 0.25]) 
        # self.state = np.random.rand(2) 
        self.doneNum = 0   
        return self.state
    
    def resetmiddle(self):
        # self.state = np.array([0.50, 0.75]) 
        # self.state = np.random.rand(2) 
        self.doneNum = 0 
        return self.state


    def render_frame(self, hw=15, imagesize=(300,300), camera_id=0, heatmap=None):
        hw = 100
        margin = 4
        screen_width = hw + margin*2
        screen_height = hw + margin*2
        frame = np.zeros([screen_width, screen_height, 3]) + 255
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
        
        minx, maxx = int((0.5-self.barw/2)*hw) + margin, int((0.5+self.barw/2)*hw) + margin 
        miny, maxy = int((0.5-self.barh)*hw) + margin, int((0.5+self.barw/2)*hw) + margin 
        for i in range(3): frame[minx:maxx, miny:maxy, i] = color[i]; 
        
        minx, maxx = int((0.5-self.barh)*hw) + margin, int((0.5+self.barw/2)*hw) + margin 
        miny, maxy = int((0.5-self.barw/2)*hw) + margin, int((0.5+self.barw/2)*hw) + margin 
        for i in range(3): frame[minx:maxx, miny:maxy, i] = color[i]; 
        
        
        return frame 

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None