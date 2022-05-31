import time
import joblib
import os
import cv2
import os.path as osp
import torch
import numpy as np
import torch.nn.functional as F 
import pybullet
#from fireup import EpochLogger
from utils.logx import EpochLogger, colorize

import visdom
import json
import random
import pybullet

from utils_ import get_SREPS, get_srep, get_transitionse

import matplotlib.pyplot as plt

# from mujoco_py import GlfwContext
# GlfwContext(offscreen=True)

def visualize_bound_box(shape_size, shape_CoM, rgba=[0,0,1,0.2]):
	boxID = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents = shape_size)
	visualShapeId = -1

	bound_box = pybullet.createMultiBody(0.01, boxID, visualShapeId, shape_size)

	# Disable Collisions with external objects
	pybullet.setCollisionFilterGroupMask(bound_box, -1, 0, 0)
	pybullet.changeVisualShape(bound_box, -1, rgbaColor=rgba)
	constraint = pybullet.createConstraint(parentBodyUniqueId=bound_box,
				parentLinkIndex=-1,
				childBodyUniqueId=-1,
				childLinkIndex=-1,
				jointType=pybullet.JOINT_FIXED,
				jointAxis=[0,0,0],
				parentFramePosition=[0,0,0],
				childFramePosition= shape_CoM)

def load_envs_agent(args, fpath, itr='last'):
 
    agent = joblib.load(osp.join(fpath, 'agent.pkl'))['agent']
    itr = agent.load_policy(fpath, itr, sord=args.policy)
    agent.change_device2cpu() 
    agent.change_device2device()
    
    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    # try:
    #     state = joblib.load(osp.join(fpath, 'vars.'+itr2+'.pkl'))
    #     env = state['env']
    # except:
    #     env = None

    import gym
    env = gym.make(args.env_name, version="GUI")
    
    print(colorize('LOADING {} {} {} {}'.format(args.env_name, args.fpath, args.policy, itr), 'red', bold=True))
    
    return env, agent

def load_config(fpath, name='config.json'):
    config = None
    try:
        config_path = open(os.path.join(fpath, name))
        config = json.load(config_path)
    except:
        print('No file named config.json')
    return config


def run_policy(env, agent, args):
    n = 0

    while n < args.episodes:
        n += 1
        o, d = env.reset(), False
        pit = 0 
        while not d:
            pit += 1
            a = agent.select_action(o)                    
            no, r, d, _ = env.step(a)
            o = no
            d = (pit>=env._max_episode_steps)
            time.sleep(1./20.) 
        pybullet.removeBody(3)
        pybullet.removeBody(4)
        pybullet.removeBody(5)

def run(vis, args):
    config = load_config(args.fpath)
    
    args.env_name = config['vas']['env']
    args.seed = config['vas']['seed']
    args.policy = config['vas']['policy'] 
    
    print('env_name: \t', args.env_name)
    
    env, agent = load_envs_agent(args, args.fpath, args.itr if args.itr >=0 else 'last')
    env.seed(args.seed)
    visualize_bound_box(shape_size=np.array([0.2,0.3,0.2]),shape_CoM=np.array([0.4,0,0.2]))
    run_policy(env, agent, args)
 
        
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--env_name', type=str, default=None)
    parser.add_argument('--seed', '-s', type=int, default=None)
    
    parser.add_argument('--len', '-l', type=int, default=250)
    parser.add_argument('--episodes', '-n', type=int, default=10)
    
    parser.add_argument('--itr', '-i', type=int, default=-1)

    args = parser.parse_args()
    
    vis = None
        
    run(vis, args)

'''
video
record
rep
'''
