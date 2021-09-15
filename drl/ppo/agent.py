import os
from collections import defaultdict

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from .nn_blocks import CategoricalActor, GaussianActor, Critic, ActorCritic


class PPOAgent(object):
    @staticmethod
    def discounted_backward_sum(v, done, k=1):
        for i in range(len(v) - 2, -1, -1):
            v[i] += (1 - done[i]) * k * v[i + 1]

    def __init__(self, args, env, device):
        self.device = device
        '''if env.discrete:
            self.actor = CategoricalActor(args, env.n_obs, env.n_act).to(device)
        else:
            self.actor = GaussianActor(args, env.n_obs, env.n_act).to(device)
        self.critic = Critic(args, env.n_obs).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.policy_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        '''
        self.policy = ActorCritic(args, env.n_obs, env.n_act, env.continuous).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.lr)
        #print(self.actor)
    
    @torch.no_grad()
    def act(self, obs, deterministic=False):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        #action = self.actor.get_action(obs, deterministic)
        action = self.policy.get_action(obs, deterministic)
        return action.cpu().numpy()
    
    def lr_decay(self, args, cur, total):
        '''for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = args.policy_lr * (1 - cur / total)
        for param_group in self.critic_optimizer.param_groups:
            param_group["lr"] = args.critic_lr * (1 - cur / total)'''
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = args.critic_lr * (1 - cur / total)
    
    def learn(self, args, buffer, last_obs, T):
        S, batch_size = args.sample_steps, args.sample_steps // args.num_minibatch
        n, L = args.num_env, S // args.num_env
        data = buffer.get(np.arange(L))

        obs = torch.as_tensor(data["obs"]).to(self.device)
        obs = obs.view(-1, *obs.shape[2:])
        #print("obs shape:", obs.size())
        #exit(0)
        with torch.no_grad():
            #value = self.critic(obs).cpu().numpy().reshape(L, n)
            value = self.policy.get_value(obs).cpu().numpy().reshape(L, n)
            last_obs = torch.tensor(last_obs, dtype=torch.float32, device=self.device)
            #last_value = self.critic(last_obs).cpu().numpy().reshape(1, n)
            last_value = self.policy.get_value(last_obs).cpu().numpy().reshape(1, n)
        #print("value 0", value.shape, last_value.shape)
        done = data["done"]
        #print("done", done.shape)
        #print("reward", data["reward"].shape)
        #print("next_V", value[1:].shape, np.concatenate([value[1:], last_value], 0).shape)
        adv = data["reward"] + args.gamma * (1 - done) * np.concatenate([value[1:], last_value], 0) - value
        self.discounted_backward_sum(adv, done, args.lam * args.gamma)
        ret = torch.tensor(value + adv, dtype=torch.float32, device=self.device)
        adv = torch.tensor(adv, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            #old_pi = self.actor(obs)
            old_pi = self.policy.get_pi(obs)
        #print("batch_shape", old_pi._batch_shape)
        action = torch.as_tensor(data["action"]).to(self.device).view(old_pi.batch_shape)
        #print("action_shape", action.shape)
        #print("log_p", old_pi.log_prob(action).shape)
        old_log_p = old_pi.log_prob(action).reshape(S, -1).sum(-1)
        #print("old_pi_loc_shape:", old_pi.loc.size())
        #print("old_log_p:", old_log_p)
        #exit(0)

        #reward = np.copy(data["reward"])
        #self.discounted_backward_sum(reward, done, args.gamma)
        #ret = torch.tensor(reward, dtype=torch.float32, device=self.device).view(-1)

        adv, ret, old_log_p = adv.view(-1), ret.view(-1), old_log_p.view(-1)
        stats = defaultdict(float)
        for _ in range(args.reuse_times):
            rand_perm = np.random.permutation(S)
            for k in stats.keys():
                if k in ["kl", "clip_fraction", "entropy"]:
                    stats[k] = 0
            for batch_start in range(0, S - batch_size + 1, batch_size):
                idx = rand_perm[batch_start:batch_start + batch_size]
                #value = self.critic(obs[idx]).reshape(-1)
                value = self.policy.get_value(obs[idx]).reshape(-1)
                value_loss = ((value - ret[idx]) ** 2).mean()

                #self.critic_optimizer.zero_grad()
                #critic_loss.backward()
                #stats["critic_grad_norm"] += clip_grad_norm_(self.critic.parameters(), 1e9).item()
                #self.critic_optimizer.step()
                stats["value_loss"] += value_loss.item()

                #pi = self.actor(obs[idx])
                pi = self.policy.get_pi(obs[idx])
                log_p = pi.log_prob(action[idx]).reshape(batch_size, -1).sum(-1)
                ratio = (log_p - old_log_p[idx]).exp()
                adv_batch = adv[idx]
                adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)
                policy_loss = -torch.min(ratio * adv_batch, ratio.clamp(1 - args.clip_ratio, 1 + args.clip_ratio) * adv_batch).mean()
                entropy_loss = -pi.entropy().mean()
                #policy_loss = -(log_p * adv).mean()

                loss = policy_loss + args.value_loss_coef * value_loss + args.entropy_coef * entropy_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                #self.actor_optimizer.zero_grad()
                #policy_loss.backward()
                #stats["actor_grad_norm"] += clip_grad_norm_(self.actor.parameters(), args.max_grad_norm).item()
                #self.actor_optimizer.step()

                stats["policy_loss"] += policy_loss.item()
                stats["kl"] += (old_log_p[idx] - log_p).mean().item()
                stats["clip_fraction"] += (ratio.gt(1 + args.clip_ratio) | ratio.lt(1 - args.clip_ratio)).to(torch.float32).mean().item()
                stats["entropy"] += pi.entropy().mean().item()

        for k in stats.keys():
            if k in ["actor_grad_norm", "critic_grad_norm", "policy_loss", "critic_loss"]:
                stats[k] /= args.reuse_times
            stats[k] /= args.num_minibatch
        if args.wandb_show:
            from .logger import log
            log("train", T, stats)
        stats["T"] = T
        return stats

    def load(self, load_file):
        return
        data = torch.load(load_file)
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])
        self.actor_optimizer.load_state_dict(data["actor_optim"])
        self.critic_optimizer.load_state_dict(data["critic_optim"])
    
    def save(self, save_dir, file_name):
        return
        data = {"actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optim": self.actor_optimizer.state_dict(),
                "critic_optim": self.critic_optimizer.state_dict()}
        torch.save(data, os.path.join(save_dir, file_name))