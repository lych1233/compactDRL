from functools import partial
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    def __init__(self, args, n_in, n_out):
        super(NoisyLinear, self).__init__()
        self.use_noise = "noisy_net" in args.enhancement
        self.noise_std = args.noise_std
        self.n_in, self.n_out = n_in, n_out
        self.w_mean = nn.Parameter(torch.zeros(n_out, n_in).uniform_(-1, 1) / np.sqrt(n_in))
        self.w_std = nn.Parameter(torch.zeros(n_out, n_in).fill_(self.noise_std / np.sqrt(n_in)))
        self.register_buffer('w_eps', torch.zeros(n_out, n_in)) # We need to store current epsilon in state_dict for complete recover
        self.b_mean = nn.Parameter(torch.zeros(n_out).uniform_(-1, 1) / np.sqrt(n_in))
        self.b_std = nn.Parameter(torch.zeros(n_out).fill_(self.noise_std / np.sqrt(n_out)))
        self.register_buffer('b_eps', torch.zeros(n_out))
        self.reset_noise()

    def _scaled_noise(self, n):
        x = torch.randn(n, device=self.w_mean.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        eps_in = self._scaled_noise(self.n_in) 
        eps_out = self._scaled_noise(self.n_out)
        self.w_eps.copy_(eps_out.ger(eps_in))
        self.b_eps.copy_(eps_out)
    
    def forward(self, x):
        if self.training and self.use_noise:
            return F.linear(x, self.w_mean + self.w_std * self.w_eps, self.b_mean + self.b_std * self.b_eps)
        else:
            return F.linear(x, self.w_mean, self.b_mean)

class MLPFeature(nn.Sequential):
    def __init__(self, input_dim, hidden_dim):
        super(MLPFeature, self).__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

class CNNFeature(nn.Sequential):
    def __init__(self, args, in_channels, width, height, channel_dim):
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
        self.conv_nodes = in_channels * width * height
        super(CNNFeature, self).__init__(*layers)

class QNet(nn.Module):
    def __init__(self, args, n_obs, n_act):
        super(QNet, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.n_act = n_act
        if len(n_obs) == 1:
            self.feature_extractor = MLPFeature(*n_obs, self.hidden_dim)
            self.feature_dim = self.hidden_dim
        else:
            self.feature_extractor = CNNFeature(args, *n_obs, args.channel_dim)
            self.feature_dim = self.feature_extractor.conv_nodes
        self.dueling = "dueling" in args.enhancement
        self.distributional = "distributional" in args.enhancement
        self.atoms = args.atoms
        self.support = torch.linspace(args.minV, args.maxV, self.atoms)
        noisy_layer = partial(NoisyLinear, args)
        self.value_mlp = nn.Sequential(OrderedDict([
            ("linear1", noisy_layer(self.feature_dim, self.hidden_dim)),
            ("relu", nn.ReLU()),
            ("linear2", noisy_layer(self.hidden_dim, self.atoms if self.distributional else 1)),
        ]))
        self.advantage_mlp = nn.Sequential(OrderedDict([
            ("linear1", noisy_layer(self.feature_dim, self.hidden_dim)),
            ("relu", nn.ReLU()),
            ("linear2", noisy_layer(self.hidden_dim, n_act * self.atoms if self.distributional else n_act)),
        ]))
    
    def to(self, device):
        self.support = self.support.to(device)
        return super(QNet, self).to(device)
    
    def forward(self, obs, get_distribution=False):
        feature = self.feature_extractor(obs)
        value = self.value_mlp(feature)
        advantage = self.advantage_mlp(feature)
        if self.distributional:
            value = value.view(-1, 1, self.atoms)
            advantage = advantage.view(-1, self.n_act, self.atoms)
        if self.dueling:
            Q = value + advantage - advantage.mean(1, keepdim=True)
        else:
            Q = advantage
        if self.distributional:
            Q = Q.softmax(2)
            if get_distribution:
                return Q
            else:
                return (Q * self.support).sum(2)
        else:
            return Q

    def reset_noise(self):
        self.value_mlp.linear1.reset_noise()
        self.value_mlp.linear2.reset_noise()
        self.advantage_mlp.linear1.reset_noise()
        self.advantage_mlp.linear2.reset_noise()
