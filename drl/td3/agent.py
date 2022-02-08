import os
import itertools
from copy import deepcopy

import numpy as np
import torch

from .nn_blocks import Actor, Critic, make_feature_extractor_fn
from .noise import GaussianNoise, OUNoise


def soft_update_net(args, online_net, target_net):
    tau = args.tau
    with torch.no_grad():
        for online_x, target_x in zip(online_net.parameters(), target_net.parameters()):
            target_x.data.mul_(1 - tau)
            target_x.data.add_(tau * online_x.data)

class TD3Agent(object):
    def __init__(self, args, env, device):
        self.device = device
        feature_extractor_fn = make_feature_extractor_fn(args, env)
        self.online_actor = Actor(args, env.n_act, feature_extractor_fn).to(device)
        self.actor_optimizer = torch.optim.Adam(self.online_actor.parameters(), lr=args.actor_lr)

        self.online_Q1, self.online_Q2 = Critic(args, env.n_act, feature_extractor_fn).to(device), Critic(args, env.n_act, feature_extractor_fn).to(device)
        self.Q_params = itertools.chain(self.online_Q1.parameters(), self.online_Q2.parameters())
        self.Q_optimizer = torch.optim.Adam(self.Q_params, lr=args.critic_lr)

        self.target_actor, self.target_Q1, self.target_Q2 = deepcopy(self.online_actor), deepcopy(self.online_Q1), deepcopy(self.online_Q2)
        for x in self.target_actor.parameters():
            x.requires_grad = False
        for x in self.target_Q1.parameters():
            x.requires_grad = False
        for x in self.target_Q2.parameters():
            x.requires_grad = False
        
        self.learn_steps = 0
        noise_gen_seed = np.random.randint(1e9)
        self.noise_gen = OUNoise(env.n_act, noise_gen_seed) if args.OU_noise else GaussianNoise(env.n_act, noise_gen_seed)
    
    @torch.no_grad()
    def act(self, obs, noise_scale=0):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.online_actor(obs).squeeze(0).cpu().numpy()
        action += noise_scale * self.noise_gen()
        return np.clip(action, -1, 1)
    
    def lr_decay(self, args, cur, total):
        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = args.actor_lr * (1 - cur / total)
        for param_group in self.Q_optimizer.param_groups:
            param_group["lr"] = args.critic_lr * (1 - cur / total)
    
    def learn(self, args, buffer, T):
        stats = {}
        batch_size = args.batch_size
        
        self.learn_steps += 1
        idx = np.random.choice(len(buffer), batch_size, replace=True)
        data = buffer.get(idx, collect_next_obs=True)
        obs = data["obs"]
        action = data["action"]
        reward = data["reward"]
        done = data["done"]
        next_obs = data["next_obs"]
        
        Q1, Q2 = self.online_Q1(obs, action), self.online_Q2(obs, action)
        with torch.no_grad():
            next_action = self.target_actor(next_obs)
            next_eps = (args.target_noise_scale * torch.randn_like(next_action)).clamp(-args.target_noise_clip, args.target_noise_clip)
            next_action = (next_action + next_eps).clamp(-1, 1)
            
            next_Q1, next_Q2 = self.target_Q1(next_obs, next_action), self.target_Q2(next_obs, next_action)
            next_Q = torch.min(next_Q1, next_Q2)
            tar_Q = reward + args.gamma * (1 - done) * next_Q
        
        Q_loss = ((Q1 - tar_Q) ** 2).mean() + ((Q2 - tar_Q) ** 2).mean()
        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()
        stats["Q1"] = Q1.mean().item()
        stats["Q2"] = Q2.mean().item()
        stats["Q_loss"] = Q_loss.item()

        if self.learn_steps % args.update_delay == 0:
            for x in self.Q_params:
                x.requires_grad = False
            
            actor_loss = -self.online_Q1(obs, self.online_actor(obs)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            stats["actor_loss"] = actor_loss.item()
            
            for x in self.Q_params:
                x.requires_grad = True
            soft_update_net(args, self.online_actor, self.target_actor)
            soft_update_net(args, self.online_Q1, self.target_Q1)
            soft_update_net(args, self.online_Q2, self.target_Q2)
        
        if args.wandb_show:
            from .logger import log
            log("train", T, stats)
        stats["T"] = T
        return stats

    def load(self, load_file):
        data = torch.load(load_file)
        self.online_actor.load_state_dict(data["online_actor"])
        self.online_Q1.load_state_dict(data["online_Q1"])
        self.online_Q2.load_state_dict(data["online_Q2"])
        self.target_actor.load_state_dict(data["target_actor"])
        self.target_Q1.load_state_dict(data["target_Q1"])
        self.target_Q2.load_state_dict(data["target_Q2"])
        self.actor_optimizer.load_state_dict(data["actor_optimizer"])
        self.Q_optimizer.load_state_dict(data["Q_optimizer"])
    
    def save(self, save_dir, file_name):
        data = {"online_actor": self.online_actor.state_dict(),
                "online_Q1": self.online_Q1.state_dict(),
                "online_Q2": self.online_Q2.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "target_Q1": self.target_Q1.state_dict(),
                "target_Q2": self.target_Q2.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "Q_optimizer": self.Q_optimizer.state_dict()}
        torch.save(data, os.path.join(save_dir, file_name))
