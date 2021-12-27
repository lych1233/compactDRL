import functools

import torch
import torch.nn as nn


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

class Actor(nn.Module):
    def __init__(self, args, n_act, feature_extractor_fn):
        super(Actor, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.feature_extractor = feature_extractor_fn()
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.fc = nn.Linear(self.hidden_dim, n_act)
    
    def forward(self, obs):
        feature = self.feature_extractor(obs)
        x = self.mlp(feature)
        return torch.tanh(self.fc(x))

class Critic(nn.Module):
    def __init__(self, args, n_act, feature_extractor_fn):
        super(Critic, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.obs_feature_extractor = feature_extractor_fn()
        self.action_feature_extractor = MLPFeature(n_act, self.hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.fc = nn.Linear(self.hidden_dim, 1)
    
    def forward(self, obs, action):
        obs_feature = self.obs_feature_extractor(obs)
        action_feature = self.action_feature_extractor(action)
        x = self.mlp(obs_feature + action_feature)
        Q = self.fc(x).squeeze(-1)
        return Q
