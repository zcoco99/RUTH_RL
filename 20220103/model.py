import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_dim, disc_obs_index=None):
        super(Discriminator, self).__init__()
        
        self.disc_obs_index = disc_obs_index
        if disc_obs_index is not None:
            num_inputs = len(disc_obs_index)

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear31 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear32 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear33 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, num_outputs)

        self.apply(weights_init_)

    def forward(self, state):
        if self.disc_obs_index is not None:
            state = state[:, self.disc_obs_index]
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        # x = F.relu(self.linear33(F.relu(self.linear32(F.relu(self.linear31(x))))))
        x = self.linear4(x)
        x = 2 * F.tanh(x) # TBD

        return x # regression label, unnormalized
    

class Predictor(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_dim, disc_obs_index=None):
        super(Predictor, self).__init__()
        
        self.disc_obs_index = disc_obs_index
        if disc_obs_index is not None:
            num_outputs = len(disc_obs_index)

        self.linear1 = nn.Linear(num_inputs+num_outputs, hidden_dim) # s_t z
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, num_outputs)
        self.log_std_linear = nn.Linear(hidden_dim, num_outputs)

        self.apply(weights_init_)

    def forward(self, context, state, next_state):
        
        if self.disc_obs_index is not None:
            state = state[:, self.disc_obs_index]
            next_state = next_state[:, self.disc_obs_index]
        xu = torch.cat([context, state], 1)
        x = F.relu(self.linear1(xu))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX) # 
        std = log_std.exp()
        std = std * 0 + (2* math.pi)**-0.5 
        normal = Normal(mean, std)
        log_prob = normal.log_prob(state) 
        return log_prob 

# class Predictor(nn.Module):
#     def __init__(self, num_inputs, num_outputs, hidden_dim, disc_obs_index=None):
#         super(Predictor, self).__init__()
        
#         self.disc_obs_index = disc_obs_index
#         if disc_obs_index is not None:
#             num_outputs = len(disc_obs_index)

#         self.linear1 = nn.Linear(num_inputs, hidden_dim)
#         self.linear2 = nn.Linear(hidden_dim, hidden_dim)
#         self.mean_linear = nn.Linear(hidden_dim, num_outputs)
#         self.log_std_linear = nn.Linear(hidden_dim, 1)

#         self.apply(weights_init_)

#     def forward(self, context, state):
        
#         if self.disc_obs_index is not None:
#             state = state[:, self.disc_obs_index]
            
#         x = F.relu(self.linear1(context))
#         x = F.relu(self.linear2(x))
#         mean = self.mean_linear(x)
#         log_std = self.log_std_linear(x)
#         log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX) # 
#         std = log_std.exp()
#         std = std * 0 + (2* math.pi)**-0.5 
#         normal = Normal(mean, std)
#         log_prob = normal.log_prob(state) 
#         return log_prob 



# A discriminator for trajectories
class DiscriminatorT(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_dim):
        super(DiscriminatorT, self).__init__()

        self.lstm = nn.LSTM(input_size=num_inputs, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_dim, num_outputs)
        
        self.apply(weights_init_)

    def forward(self, seq):
        x, _ = self.lstm(seq)
        x = self.linear(x)
        x = torch.mean(x, dim=1) # average over all sample points on one trajectory

        return x
    
    

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        
        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        
        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        
        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.softplus(self.linear1(state))
        x = F.softplus(self.linear2(x))
        x = F.softplus(self.linear3(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)
    
    


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, num_actions)
        
        
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
            
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
    
    
