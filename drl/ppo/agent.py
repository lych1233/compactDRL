import os

import torch

from .nn_blocks import CategoricalActor, GaussianActor, Critic


class PPOAgent(object):
    def __init__(self, args, env, device):
        self.device = device
        if env.discrete:
            self.actor = CategoricalActor(args, env.n_obs, env.n_act).to(device)
        else:
            self.actor = GaussianActor(args, env.n_obs, env.n_act).to(device)
        self.critic = Critic(args, env.n_obs).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.policy_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)
    
    @torch.no_grad()
    def act(self, obs, deterministic=False):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action = self.actor.get_action(obs.unsqueeze(0), deterministic)
        return action.item()

    def load(self, load_file):
        data = torch.load(load_file)
        self.actor.load_state_dict(data['actor'])
        self.critic.load_state_dict(data['critic'])
        self.actor_optimizer.load_state_dict(data['actor_optim'])
        self.critic_optimizer.load_state_dict(data['critic_optim'])
    
    def save(self, save_dir, file_name):
        data = {'actor', self.actor.state_dict(),
                'critic', self.critic.state_dict(),
                'actor_optim', self.actor_optimizer.state_dict(),
                'critic_optim', self.critic_optimizer.state_dict()}
        torch.save(data, os.path.join(save_dir, file_name))