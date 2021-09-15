import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


class MLPFeature(nn.Sequential):
    def __init__(self, input_dim, hidden_dim):
        super(MLPFeature, self).__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
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

class CategoricalActor(nn.Module):
    def __init__(self, args, n_act):
        super(CategoricalActor, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.fc = nn.Linear(self.hidden_dim, n_act)
        #for w in self.fc.parameters():
        #    w.data.mul_(0.01)
    
    def forward(self, feature):
        logits = self.fc(feature)
        return Categorical(logits=logits)
    
    def get_action(self, feature, deterministic=False):
        pi = self.forward(feature)
        if deterministic:
            return pi.logits.argmax(-1)
        else:
            return pi.sample()

class GaussianActor(nn.Module):
    def __init__(self, args, n_act):
        super(GaussianActor, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.fc = nn.Linear(self.hidden_dim, n_act)
        self.log_std = nn.Parameter(-0.5 * torch.ones(n_act))
    
    def forward(self, feature):
        mean, std = self.fc(feature), torch.exp(self.log_std)
        return Normal(mean, std)
    
    def get_action(self, feature, deterministic=False):
        pi = self.forward(feature)
        if deterministic:
            return pi.mean
        else:
            return pi.sample()

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.fc = nn.Linear(self.hidden_dim, 1)
    
    def forward(self, feature):
        value = self.fc(feature)
        return value

class ActorCritic(nn.Module):
    def __init__(self, args, n_obs, n_act, is_continuous):
        super(ActorCritic, self).__init__()
        self.hidden_dim = args.hidden_dim
        if isinstance(n_obs, int):
            self.share_feature = False
            self.actor_feature_extractor = MLPFeature(n_obs, self.hidden_dim)
            self.critic_feature_extractor = MLPFeature(n_obs, self.hidden_dim)
        else:
            self.share_feature = True
            self.share_feature_extractor = CNNFeature(args, *n_obs, args.channel_dim, self.hidden_dim)
        self.actor = GaussianActor(args, n_act) if is_continuous else CategoricalActor(args, n_act)
        self.critic = Critic(args)
    
    def get_pi(self, obs):
        if self.share_feature:
            feature = self.share_feature_extractor(obs)
        else:
            feature = self.actor_feature_extractor(obs)
        return self.actor(feature)
    
    def get_action(self, obs, deterministic=False):
        if self.share_feature:
            feature = self.share_feature_extractor(obs)
        else:
            feature = self.actor_feature_extractor(obs)
        return self.actor.get_action(feature, deterministic)
    
    def get_value(self, obs):
        if self.share_feature:
            feature = self.share_feature_extractor(obs)
            return self.critic(feature)
        else:
            feature = self.critic_feature_extractor(obs)
            return self.critic(feature)
