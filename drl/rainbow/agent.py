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
        self.gamma_powers = (args.gamma ** torch.arange(args.multi_step)).to(device)
    
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
    
    def learn(self, args, buffer, pool, T):
        stats = {}
        batch_size = args.batch_size
        
        idx, importance_factor = pool.sample(batch_size)
        importance_factor = torch.FloatTensor(importance_factor).to(self.device)

        data = buffer.get(idx)
        obs = torch.as_tensor(data["obs"]).to(self.device)
        action = torch.as_tensor(data["action"]).to(self.device)
        next_data = buffer.get(idx + args.multi_step)
        next_obs = torch.as_tensor(next_data["obs"]).to(self.device)
        
        all_data = buffer.get(np.expand_dims(idx, 1) + np.arange(args.multi_step))
        reward = torch.as_tensor(all_data["reward"]).to(self.device)
        done = torch.as_tensor(all_data["done"]).to(self.device)
        mask = done.to(torch.bool)
        for i in range(1, args.multi_step, 1):
            mask[:, i].logical_or_(mask[:, i - 1])
        
        online_Q = self.online_net(obs).gather(1, action.view(-1, 1)).view(-1)
        with torch.no_grad():
            next_Q = self.target_net(next_obs).max(1)[0]
            target_Q = reward + args.gamma * (1 - done) * next_Q
        Q_loss = ((online_Q - target_Q) ** 2).mean()
        self.optimizer.zero_grad()
        Q_loss.backward()
        stats["Q_grad_norm"] = clip_grad_norm_(self.online_net.parameters(), args.max_grad_norm).item()
        self.optimizer.step()
        stats["Q"] = online_Q.mean().item()
        stats["Q_loss"] = Q_loss.item()

        self.update_times += 1
        if self.update_times % args.target_update_interval:
            self.target_net.load_state_dict(self.online_net.state_dict()) 
        
        if args.wandb_show:
            from .logger import log
            log("train", T, stats)
        stats["T"] = T
        return stats

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
