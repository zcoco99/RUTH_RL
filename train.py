# LRoS implementation 
import joblib
import numpy as np 
import torch
import gym 
import time

from utils.mpi_tools import mpi_fork, proc_id, mpi_statistics_scalar, num_procs
from utils.logx import EpochLogger

import os.path as osp
import random
import visdom
import pybullet

from sac import SAC, count_vars
from replay_memory import ReplayMemory


torch.autograd.set_detect_anomaly(True)

def lros(vis, env, args, logger_kwargs=dict(), save_freq=5):
    
    logger = EpochLogger(**logger_kwargs)
    
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)

#    env.seed(args.seed)
    # env.reset()
    
    # _, _, _, info = env.step(env.action_space.sample())
    vas = vars(args)
    logger.save_config(locals())
    logger.add_vis(vis)
    obs_dim = env.observation_space.shape[0]
#    obs_dim_dc = env.observation_space_disc.shape[0] if env.observation_space_disc is not None else obs_dim
    
    agent = SAC(obs_dim, env.action_space, args)
    
    logger.save_agent({'agent': agent})
    if args.ctdir is not None: 
        agent.load_disc(args.ctdir, 'last')
        import json, os; 
        tempconfig = json.load(open(os.path.join(args.ctdir, 'config.json'))) 
        agent.load_policy(args.ctdir, 'last', sord=tempconfig['vas']['policy'])
    agent.change_device2device() 
        
    
    local_episodes_per_epoch = args.episodes
    
    # Memory
    memorytheta = ReplayMemory(args.replay_size)
    memorymu = ReplayMemory(args.replay_size)
    memoryt = ReplayMemory(args.replay_size)
    
    var_counts = tuple(count_vars(module) for module in
        [agent.policy, agent.critic])
    
    logger.log('\nNumber of parameters: \t policy: %d, \t critic: %d \n'%var_counts)
    
    start_time = time.time()
    total_t = 0
    upnum = 0
    
    allState = []
    for i in range(6):
        for j in range(6):
            allState.append([i/8.0, j/8.0])
    allState = np.array(allState)
    
    
    logger_theta_mu = False
    logger_mu_t = False
    memory = ReplayMemory(args.replay_size)
    
    for epoch in range(args.epochs):
        episode_st, episode_rd,  = 0, 0
        
        for e in range(local_episodes_per_epoch):
            
            pit, o, d =0, env.reset(), False
            while not d:
                pit += 1 
                a = agent.select_action(o)
                no, r, d, info = env.step(a)
                mask = 1 if pit == env._max_episode_steps else float(not d)
                memory.push(o, a, r, no, mask)
                o = no
                
                total_t += 1 
                episode_st += 1 
                episode_rd += r 
                
                
                trainstep = 2
                if len(memory) > args.batch_size: # and total_t % trainstep == 0: # 
                    for i in range(args.updates_per_step):
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = \
                            agent.update_parameters(memory, args.batch_size, upnum)
                    logger.store(Policy=policy_loss, Critic1=critic_1_loss, \
                                Critic2=critic_2_loss, Ent=ent_loss, Alpha=alpha, \
                                Reward=r)
                    logger_theta_mu = True
                    upnum += 1
            pybullet.removeBody(2)
            pybullet.removeBody(3)
            pybullet.removeBody(4)
                    
            
            
        if proc_id()==0 and logger_theta_mu:
            logger.log_tabular('Epoch', epoch)
            
            logger.log_tabular('Policy', average_only=True)
            logger.log_tabular('Critic1', average_only=True)
            logger.log_tabular('Critic2', average_only=True)
            logger.log_tabular('Reward', average_only=True)
            logger.log_tabular('Ent', average_only=True)
            logger.log_tabular('Alpha', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()
        
        
        if proc_id()==0 and (epoch % save_freq == 0) or (epoch == args.epochs - 1):
            agent.change_device2cpu()
            logger.save_state([agent.policy, 
                        agent.critic], 
                        epoch)
            agent.change_device2device()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser() # 
    parser.add_argument('--env', type=str, default='kelin-v0') 
    parser.add_argument('--seed', '-s', type=int, default=0)
    
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.05, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--updates_per_step_cla', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    
    parser.add_argument('--start_steps', type=int, default=2, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=75000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=2048)
    
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--sub_exp_name', type=str, default=None)
    parser.add_argument('--ctdir', type=str, default=None)
    
    args = parser.parse_args()
    
    
    args.exp_name = ''+args.env
    
    PREF = 'seed-'
    vis = None
    vis = visdom.Visdom(env=PREF+args.exp_name+'-'+str(args.seed)+'')
    if vis is None:
        PREF = 'None-'
        args.batch_size = 16 
        args.hidden_size = 2 
    args.sub_exp_name = PREF 
    mpi_fork(args.cpu)

    from utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.sub_exp_name, args.seed)
    
    lros(vis, gym.make(args.env, version="GUI"), args, 
         logger_kwargs=logger_kwargs)
