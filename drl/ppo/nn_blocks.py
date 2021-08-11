import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


class MLPFeature(nn.Sequential):
    def __init__(self, input_dim, hidden_nodes):
        super(MLPFeature, self).__init__(
            nn.Linear(input_dim, hidden_nodes),
            nn.ReLU(),
        )
        self.output_dim = hidden_nodes

class CNNFeature(nn.Sequential):
    def __init__(self, args, in_channels, width, height, hidden_channels):
        divider_list = list(map(int, args.channel_divider.split(",")))
        kernel_list = list(map(int, args.kernel_size.split(",")))
        stride_list = list(map(int, args.stride.split(",")))
        layers = []
        for divider, kernel, stride in zip(divider_list, kernel_list, stride_list):
            layers.append(nn.Conv2d(in_channels, hidden_channels // divider, kernel, stride))
            layers.append(nn.ReLU())
            in_channels = hidden_channels // divider
            width = (width - kernel) // stride + 1
            height = (height - kernel) // stride + 1
        layers.append(nn.Flatten())
        super(CNNFeature, self).__init__(*layers)
        self.output_dim = in_channels * width * height

class CategoricalActor(nn.Module):
    def __init__(self, args, n_obs, n_act):
        super(CategoricalActor, self).__init__()
        self.hidden_dim = args.hidden_dim
        if isinstance(n_obs, int):
            self.feature_extractor = MLPFeature(n_obs, self.hidden_dim)
        else:
            self.feature_extractor = CNNFeature(args, *n_obs, self.hidden_dim)
        self.feature_dim = self.feature_extractor.output_dim
        self.fc = nn.Linear(self.feature_dim, n_act)
    
    def forward(self, obs):
        feature = self.feature_extractor(obs)
        logits = self.fc(feature)
        return Categorical(logits=logits)
    
    def get_action(self, obs, deterministic=False):
        pi = self.forward(obs)
        if deterministic:
            return pi.probs.argmax(-1)
        else:
            return pi.sample()

class GaussianActor(nn.Module):
    def __init__(self, args, n_obs, n_act):
        super(GaussianActor, self).__init__()
        self.hidden_dim = args.hidden_dim
        if isinstance(n_obs, int):
            self.feature_extractor = MLPFeature(n_obs, self.hidden_dim)
        else:
            self.feature_extractor = CNNFeature(args, *n_obs, self.hidden_dim)
        self.feature_dim = self.feature_extractor.output_dim
        self.fc = nn.Linear(self.feature_dim, n_act)
        self.log_std = nn.Parameter(-0.5 * torch.ones(n_act))
    
    def forward(self, obs):
        feature = self.feature_extractor(obs)
        mean, std = self.fc(feature), torch.exp(self.log_std)
        return Normal(mean, std)
    
    def get_action(self, obs, deterministic=False):
        pi = self.forward(obs)
        if deterministic:
            return pi.mean
        else:
            return pi.sample()

class Critic(nn.Module):
    def __init__(self, args, n_obs):
        super(Critic, self).__init__()
        self.hidden_dim = args.hidden_dim
        if isinstance(n_obs, int):
            self.feature_extractor = MLPFeature(n_obs, self.hidden_dim)
        else:
            self.feature_extractor = CNNFeature(args, *n_obs, self.hidden_dim)
        self.feature_dim = self.feature_extractor.output_dim
        self.fc = nn.Linear(self.feature_dim, 1)
    
    def forward(self, obs):
        feature = self.feature_extractor(obs)
        value = self.fc(feature)
        return value
