import os
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

class DDPGAgent(object):
    def __init__(self, args, env, device):
        self.device = device
        feature_extractor_fn = make_feature_extractor_fn(args, env)
        self.online_actor = Actor(args, env.n_act, feature_extractor_fn).to(device)
        self.actor_optimizer = torch.optim.Adam(self.online_actor.parameters(), lr=args.actor_lr)

        self.online_critic = Critic(args, env.n_act, feature_extractor_fn).to(device)
        self.critic_optimizer = torch.optim.Adam(self.online_critic.parameters(), lr=args.critic_lr)

        self.target_actor, self.target_critic = deepcopy(self.online_actor), deepcopy(self.online_critic)
        for x in self.target_actor.parameters():
            x.requires_grad = False
        for x in self.target_critic.parameters():
            x.requires_grad = False
        
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
        for param_group in self.critic_optimizer.param_groups:
            param_group["lr"] = args.critic_lr * (1 - cur / total)
    
    def learn(self, args, buffer, T):
        stats = {}
        batch_size = args.batch_size
        
        idx = np.random.choice(len(buffer), batch_size, replace=True)
        data = buffer.get(idx, collect_next_obs=True)
        obs = data["obs"]
        action = data["action"]
        reward = data["reward"]
        done = data["done"]
        next_obs = data["next_obs"]
        
        online_Q = self.online_critic(obs, action)
        with torch.no_grad():
            next_action = self.target_actor(next_obs)
            next_Q = self.target_critic(next_obs, next_action)
            target_Q = reward + args.gamma * (1 - done) * next_Q            
        Q_loss = ((online_Q - target_Q) ** 2).mean()
        self.critic_optimizer.zero_grad()
        Q_loss.backward()
        self.critic_optimizer.step()
        stats["Q"] = online_Q.mean().item()
        stats["Q_loss"] = Q_loss.item()

        for x in self.online_critic.parameters():
            x.requires_grad = False
        
        actor_loss = -self.online_critic(obs, self.online_actor(obs)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        stats["actor_loss"] = actor_loss.item()
        
        for x in self.online_critic.parameters():
            x.requires_grad = True
        soft_update_net(args, self.online_actor, self.target_actor)
        soft_update_net(args, self.online_critic, self.target_critic)
        
        if args.wandb_show:
            from .logger import log
            log("train", T, stats)
        stats["T"] = T
        return stats

    def load(self, load_file):
        data = torch.load(load_file)
        self.online_actor.load_state_dict(data["online_actor"])
        self.online_critic.load_state_dict(data["online_critic"])
        self.target_actor.load_state_dict(data["target_actor"])
        self.target_critic.load_state_dict(data["target_critic"])
        self.actor_optimizer.load_state_dict(data["actor_optimizer"])
        self.critic_optimizer.load_state_dict(data["critic_optimizer"])
    
    def save(self, save_dir, file_name):
        data = {"online_actor": self.online_actor.state_dict(),
                "online_critic": self.online_critic.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict()}
        torch.save(data, os.path.join(save_dir, file_name))
