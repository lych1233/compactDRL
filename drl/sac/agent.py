import os
from copy import deepcopy

import numpy as np
import torch

from .nn_blocks import GaussianActor, DoubleQNet, make_feature_extractor_fn
from .noise import OUNoise


def soft_update_net(args, online_net, target_net):
    tau = args.tau
    with torch.no_grad():
        for online_x, target_x in zip(online_net.parameters(), target_net.parameters()):
            target_x.data.mul_(1 - tau)
            target_x.data.add_(tau * online_x.data)

class SACAgent(object):
    def __init__(self, args, env, device):
        self.device = device
        feature_extractor_fn = make_feature_extractor_fn(args, env)
        self.actor = GaussianActor(args, env.n_act, feature_extractor_fn).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr)

        self.online_doubleQ = DoubleQNet(args, env.n_act, feature_extractor_fn).to(device)
        self.Q_optimizer = torch.optim.Adam(self.online_doubleQ.parameters(), lr=args.lr)

        self.target_doubleQ = deepcopy(self.online_doubleQ)
        for x in self.target_doubleQ.parameters():
            x.requires_grad = False
        
        if args.additional_OU_noise > 0:
            self.addional_noise = True
            self.online_noise_scale = args.additional_OU_noise
            self.noise_gen = OUNoise(env.n_act, np.random.randint(1e9))
        else:
            self.addional_noise = False
    
    @torch.no_grad()
    def act(self, obs, deterministic=False):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(obs, deterministic).squeeze(0).cpu().numpy()
        if self.addional_noise:
            action += self.online_noise_scale * self.noise_gen()
            action = np.clip(action, -1, 1)
        return action
    
    def lr_decay(self, args, cur, total):
        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = args.lr * (1 - cur / total)
        for param_group in self.Q_optimizer.param_groups:
            param_group["lr"] = args.lr * (1 - cur / total)
    
    def learn(self, args, buffer, T):
        stats = {}
        batch_size = args.batch_size
        
        idx = np.random.choice(len(buffer), batch_size, replace=True)
        data = buffer.get(idx, collect_next_obs=True)
        obs = torch.as_tensor(data["obs"]).to(self.device)
        action = torch.as_tensor(data["action"]).to(self.device)
        reward = torch.as_tensor(data["reward"]).to(self.device)
        done = torch.as_tensor(data["done"]).to(self.device)
        next_obs = torch.as_tensor(data["next_obs"]).to(self.device)
        
        Q1, Q2 = self.online_doubleQ(obs, action)
        with torch.no_grad():
            next_action, next_log_p = self.actor(next_obs, with_log_p=True)                
            next_Q1, next_Q2 = self.target_doubleQ(next_obs, next_action)
            next_Q = torch.min(next_Q1, next_Q2)
            tar_Q = reward + args.gamma * (1 - done) * (next_Q - args.alpha * next_log_p)
        
        Q_loss = ((Q1 - tar_Q) ** 2).mean() + ((Q2 - tar_Q) ** 2).mean()
        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()
        stats["Q1"] = Q1.mean().item()
        stats["Q2"] = Q2.mean().item()
        stats["Q_loss"] = Q_loss.item()

        for x in self.online_doubleQ.parameters():
            x.requires_grad = False
        
        r_action, log_p = self.actor(obs, with_log_p=True)
        r_Q1, r_Q2 = self.online_doubleQ(obs, r_action)
        actor_loss = (args.alpha * log_p - torch.min(r_Q1, r_Q2)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        stats["actor_loss"] = actor_loss.item()
        
        for x in self.online_doubleQ.parameters():
            x.requires_grad = True
        soft_update_net(args, self.online_doubleQ, self.target_doubleQ)
        
        if args.wandb_show:
            from .logger import log
            log("train", T, stats)
        stats["T"] = T
        return stats

    def load(self, load_file):
        data = torch.load(load_file)
        self.actor.load_state_dict(data["actor"])
        self.online_doubleQ.load_state_dict(data["online_doubleQ"])
        self.target_doubleQ.load_state_dict(data["target_doubleQ"])
        self.actor_optimizer.load_state_dict(data["actor_optimizer"])
        self.Q_optimizer.load_state_dict(data["Q_optimizer"])
    
    def save(self, save_dir, file_name):
        data = {"actor": self.actor.state_dict(),
                "online_doubleQ": self.online_doubleQ.state_dict(),
                "target_doubleQ": self.target_doubleQ.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "Q_optimizer": self.Q_optimizer.state_dict()}
        torch.save(data, os.path.join(save_dir, file_name))
