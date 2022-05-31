import os
import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD, RMSprop
from utils_ import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy
import random
import numpy as np


class SAC(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        
        self.device = torch.device("cuda" if args.cuda else "cpu") 

        self.critic = QNetwork(num_inputs, \
                    action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, \
                    action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        
        if self.policy_type == "Gaussian":
            # Target Entropy = ?dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning == True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, \
                        action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, \
                        action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
            
        self.density_mu_xyz = None 

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    
    

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        
        
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        
        
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value) # JQ = ??(st,at)~D[0.5(Q1(st,at) - r(st,at) - ¦Ã(??st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value) # JQ = ??(st,at)~D[0.5(Q1(st,at) - r(st,at) - ¦Ã(??st+1~p[V(st+1)]))^2]

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # J¦Ð = ??st¡«D,¦Åt¡«N[¦Á * log¦Ð(f(¦Åt;st)|st) ? Q(st,f(¦Åt;st))]
        
        

        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs
        
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
    
    
    
    def load_policy(self, fpath, itr='last', sord='Gaussian', onlytheta=False):
        PName = sord + 'Policy.'
        if itr=='last':
            saves = [int(x.split('.')[1]) for x in os.listdir(fpath) if PName in x]
            itr = '%d'%max(saves) if len(saves) > 0 else ''
        else:
            itr = '%d'%itr
        print('loaded: ', os.path.join(fpath, PName+itr+'.pt'))
        policy = torch.load(os.path.join(fpath, PName+itr+'.pt'))
        hard_update(self.policy, policy)
        self.policy.eval()
        if onlytheta:
            self.policy.uniform_mu() 
            
        return itr
    
    def load_qnet(self, fpath, itr='last'):
        if itr=='last':
            saves = [int(x.split('.')[1]) for x in os.listdir(fpath) if 'QNetwork.' in x]
            itr = '%d'%max(saves) if len(saves) > 0 else ''
        else:
            itr = '%d'%itr
        print('loaded: ', os.path.join(fpath, 'QNetwork.'+itr+'.pt'))
        qnet = torch.load(os.path.join(fpath, 'QNetwork.'+itr+'.pt'))
        hard_update(self.critic, qnet)
        hard_update(self.critic_target, qnet)
        self.critic.eval()
        self.critic_target.eval()
        return itr
    
    def change_device2cpu(self):
        self.policy.cpu()
        self.critic.cpu()
        self.critic_target.cpu()
        
    def change_device2device(self):
        self.policy.to(device=self.device)
        self.critic.to(device=self.device)
        self.critic_target.to(device=self.device)
    

        
def count_vars(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

from utils.mpi_tools import mpi_avg
#from utils.mpi_tools import proc_id

def average_param(param):
    for p in param:
#        print(proc_id(), p.data.shape)
        p.data.copy_(torch.Tensor(mpi_avg(p.data.numpy())))

        
        
        
        