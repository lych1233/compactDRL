import os
from copy import deepcopy

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from .nn_blocks import QNet


class RainbowAgent(object):
    def __init__(self, args, env, device):
        self.device = device
        self.n_act = env.n_act
        self.online_net = QNet(args, env.n_obs, env.n_act).to(device)
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=args.lr)
        self.target_net = deepcopy(self.online_net)
        for x in self.target_net.parameters():
            x.requires_grad = False
        
        self.update_times = 0
        self.gamma_powers = (args.gamma ** torch.arange(args.n_steps)).to(device)
        self.support = torch.linspace(args.minV, args.maxV, args.atoms).to(device)
    
    @torch.no_grad()
    def act(self, obs, eps=0):
        if np.random.rand() < eps:
            return np.random.randint(0, self.n_act, (1, ))
        else:
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            Q = self.online_net(obs)
            return Q.argmax().cpu().numpy()
    
    def lr_decay(self, args, cur, total):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = args.lr * (1 - cur / total)
    
    def get_normal_td_error(self, args, obs, action, reward, done, next_obs):
        online_Q = self.online_net(obs).gather(1, action.view(-1, 1)).view(-1)
        with torch.no_grad():
            avg_Q = online_Q.mean().item()
            if "double" in args.enhancement:
                next_action = self.online_net(next_obs).argmax(1)
                next_Q = self.target_net(next_obs).gather(1, next_action.view(-1, 1)).view(-1)
            else:
                next_Q = self.target_net(next_obs).max(1)[0]
            target_Q = reward + (args.gamma ** args.n_steps) * (1 - done) * next_Q
        loss = (online_Q - target_Q) ** 2
        return loss, avg_Q
    
    def get_distributional_loss(self, args, obs, action, reward, done, next_obs):
        batch_size = args.batch_size
        online_distribution =  self.online_net(obs, get_distribution=True)[np.arange(batch_size), action]
        with torch.no_grad():
            avg_Q = (online_distribution * self.support).sum(1).mean().item()
            
            if "double" in args.enhancement:
                next_action = self.online_net(next_obs).argmax(1)
            else:
                next_action = self.target_net(next_obs).argmax(1)
            next_distribution = self.target_net(next_obs, get_distribution=True)[np.arange(batch_size), next_action]
            target_support = reward.view(-1, 1) + (args.gamma ** args.n_steps) * (1 - done).view(-1, 1) * self.support
            target_support = target_support.clamp(args.minV, args.maxV)
            
            k = (target_support - args.minV) / ((args.maxV - args.minV) / (args.atoms - 1))
            l, r = k.floor().to(torch.long), k.ceil().to(torch.long)
            l[(l > 0) * (l == r)] -= 1
            r[(r + 1 < args.atoms) * (l == r)] += 1
            offset = (torch.arange(batch_size) * args.atoms).to(k.device).view(-1, 1)
            l, r, k = (l + offset).view(-1), (r + offset).view(-1), (k + offset).view(-1)
            target_distribution = torch.zeros_like(next_distribution)
            target_distribution.view(-1).index_add_(0, l, next_distribution.view(-1) * (r - k))    
            target_distribution.view(-1).index_add_(0, r, next_distribution.view(-1) * (k - l))
        loss = -(target_distribution * online_distribution.clamp(1e-30).log()).sum(1)
        return loss, avg_Q
    
    def learn(self, args, buffer, pool, T):
        self.train()

        stats = {}
        batch_size = args.batch_size
        
        idx, importance_factor = pool.sample(batch_size)
        importance_factor = torch.FloatTensor(importance_factor).to(self.device)

        data = buffer.get(idx, terms=("obs", "action"))
        obs = data["obs"]
        action = data["action"]
        
        next_data = buffer.get(idx + args.n_steps, terms=("obs", ))
        next_obs = next_data["obs"]
        
        duration_data = buffer.get(np.expand_dims(idx, 1) + np.arange(args.n_steps), terms=("reward", "done"))
        reward = duration_data["reward"]
        done = duration_data["done"]
        for i in range(args.n_steps - 1):
            done[done[:, i] > 0.5, i + 1] = 1
        reward = reward[:, 0] + (self.gamma_powers[1:] * (1 - done[:, :-1]) * reward[:, 1:]).sum(1)
        done = done[:, -1]
    
        if "distributional" in args.enhancement:
            loss, avg_Q = self.get_distributional_loss(args, obs, action, reward, done, next_obs)
        else:
            loss, avg_Q = self.get_normal_td_error(args, obs, action, reward, done, next_obs)
        stats["Q"] = avg_Q
        if "prioritized_replay" in args.enhancement:
            pool.update_priority(idx, loss.detach().cpu().numpy())
        
        loss = (importance_factor * loss).mean()
        self.optimizer.zero_grad()
        loss.backward()
        stats["loss"] = loss.item()
        stats["loss_grad_norm"] = clip_grad_norm_(self.online_net.parameters(), args.max_grad_norm).item()
        self.optimizer.step()

        self.update_times += 1
        if self.update_times % args.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict()) 
        
        if args.wandb_show:
            from .logger import log
            log("train", T, stats)
        stats["T"] = T
        return stats
    
    def reset_noise(self):
        self.online_net.reset_noise()
        self.target_net.reset_noise()

    def train(self):
        self.online_net.train()
        self.target_net.train()

    def eval(self):
        self.online_net.eval()
        self.target_net.eval()
    
    def load(self, load_file):
        data = torch.load(load_file)
        self.online_net.load_state_dict(data["online_net"])
        self.target_net.load_state_dict(data["target_net"])
        self.optimizer.load_state_dict(data["optimizer"])
    
    def save(self, save_dir, file_name):
        data = {"online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict()}
        torch.save(data, os.path.join(save_dir, file_name))
