import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class MLPFeature(nn.Sequential):
    def __init__(self, input_dim, hidden_dim):
        super(MLPFeature, self).__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

class CNNFeature(nn.Sequential):
    def __init__(self, args, in_channels, width, height, channel_dim, hidden_dim):
        divider_list = args.channel_divider
        kernel_list = args.kernel_size
        stride_list = args.stride
        layers = []
        for divider, kernel, stride in zip(divider_list, kernel_list, stride_list):
            layers.append(nn.Conv2d(in_channels, channel_dim // divider, kernel, stride))
            layers.append(nn.ReLU())
            in_channels = channel_dim // divider
            width = (width - kernel) // stride + 1
            height = (height - kernel) // stride + 1
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_channels * width * height, hidden_dim))
        layers.append(nn.ReLU())
        super(CNNFeature, self).__init__(*layers)

def make_feature_extractor_fn(args, env):
    n_obs = env.n_obs
    if len(n_obs) == 1:
        return functools.partial(MLPFeature, *n_obs, args.hidden_dim)
    else:
        return functools.partial(CNNFeature, args, *n_obs, args.channel_dim, args.hidden_dim)

class GaussianActor(nn.Module):
    def __init__(self, args, n_act, feature_extractor_fn):
        super(GaussianActor, self).__init__()
        self.log_std_min, self.log_std_max = args.log_std_min, args.log_std_max
        self.hidden_dim = args.hidden_dim
        self.feature_extractor = feature_extractor_fn()
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(self.hidden_dim, n_act)
        self.log_std = nn.Linear(self.hidden_dim, n_act)
    
    def forward(self, obs, deterministic=False, with_log_p=False):
        feature = self.feature_extractor(obs)
        hidden = self.mlp(feature)
        mean, log_std = self.mean(hidden), self.log_std(hidden)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        pi = Normal(mean, std)
        
        action = mean if deterministic else pi.rsample()
        if with_log_p:
            log_p = pi.log_prob(action) - 2 * (np.log(2) - action - F.softplus(-2 * action))
            return torch.tanh(action), log_p.sum(-1)
        else:
            return torch.tanh(action)

class QNet(nn.Module):
    def __init__(self, args, n_act, feature_extractor_fn):
        super(QNet, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.n_act = n_act
        self.feature_extractor = feature_extractor_fn()
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim + self.n_act, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.fc = nn.Linear(self.hidden_dim, 1)
    
    def forward(self, obs, action):
        feature = self.feature_extractor(obs)
        x = self.mlp(torch.cat([feature, action], 1))
        Q = self.fc(x).squeeze(-1)
        return Q

class DoubleQNet(nn.Module):
    def __init__(self, args, n_act, feature_extractor_fn):
        super(DoubleQNet, self).__init__()
        self.q1, self.q2 = QNet(args, n_act, feature_extractor_fn), QNet(args, n_act, feature_extractor_fn)
    
    def forward(self, obs, action):
        return self.q1(obs, action), self.q2(obs, action)
