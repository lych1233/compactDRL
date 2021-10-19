import os
from collections import defaultdict

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from .nn_blocks import ActorCritic


class PPOAgent(object):
    @staticmethod
    def discounted_backward_sum(v, done, k=1):
        for i in range(len(v) - 2, -1, -1):
            v[i] += (1 - done[i]) * k * v[i + 1]

    def __init__(self, args, env, device):
        self.device = device
        self.policy = ActorCritic(args, env.n_obs, env.n_act, env.continuous).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.lr)
    
    @torch.no_grad()
    def act(self, obs, deterministic=False):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action = self.policy.get_action(obs, deterministic)
        return action.cpu().numpy()
    
    def lr_decay(self, args, cur, total):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = args.lr * (1 - cur / total)
    
    def batch_update(self, args, obs, ret, adv, action, old_log_p):
        value = self.policy.get_value(obs).reshape(-1)
        value_loss = ((value - ret) ** 2).mean()
        
        pi = self.policy.get_pi(obs)
        log_p = pi.log_prob(action).reshape(len(ret), -1).sum(-1)
        ratio = (log_p - old_log_p).exp()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        surr1 = ratio * adv
        surr2 = ratio.clamp(1 - args.clip_ratio, 1 + args.clip_ratio) * adv
        policy_loss = -torch.min(surr1, surr2).mean()
        entropy = pi.entropy().mean()

        loss = policy_loss + args.value_loss_coef * value_loss + args.entropy_coef * -entropy
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm_(self.policy.parameters(), args.max_grad_norm)
        self.optimizer.step()

        kl = (old_log_p - log_p).mean().item()
        clip_fraction = (ratio.gt(1 + args.clip_ratio) | ratio.lt(1 - args.clip_ratio)).to(torch.float32).mean().item()
        return policy_loss.item(), value_loss.item(), entropy.item(), grad_norm.item(), kl, clip_fraction
    
    def learn(self, args, buffer, last_obs, T):
        S, batch_size = args.sample_steps, args.sample_steps // args.num_minibatch
        n, L = args.num_env, S // args.num_env
        data = buffer.get(np.arange(L))

        obs = torch.as_tensor(data["obs"]).to(self.device)
        obs = obs.view(-1, *obs.shape[2:])
        with torch.no_grad():
            value = self.policy.get_value(obs).cpu().numpy().reshape(L, n)
            last_obs = torch.tensor(last_obs, dtype=torch.float32, device=self.device)
            last_value = self.policy.get_value(last_obs).cpu().numpy().reshape(1, n)
        done = data["done"]
        adv = data["reward"] + args.gamma * (1 - done) * np.concatenate([value[1:], last_value], 0) - value
        self.discounted_backward_sum(adv, done, args.lam * args.gamma)
        ret = torch.tensor(value + adv, dtype=torch.float32, device=self.device)
        adv = torch.tensor(adv, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            old_pi = self.policy.get_pi(obs)
        action = torch.as_tensor(data["action"]).to(self.device).view(old_pi.batch_shape)
        old_log_p = old_pi.log_prob(action).reshape(S, -1).sum(-1)

        adv, ret, old_log_p = adv.view(-1), ret.view(-1), old_log_p.view(-1)
        stats = defaultdict(float)
        for _ in range(args.reuse_times):
            rand_perm = np.random.permutation(S)
            for batch_start in range(0, S - batch_size + 1, batch_size):
                idx = rand_perm[batch_start:batch_start + batch_size]
                policy_loss, value_loss, entropy, grad_norm, kl, clip_fraction = self.batch_update(args, obs[idx], ret[idx], adv[idx], action[idx], old_log_p[idx])
                stats["policy_loss"] += policy_loss / args.reuse_times
                stats["value_loss"] += value_loss / args.reuse_times
                stats["grad_norm"] += grad_norm / args.reuse_times
                if _ + 1 == args.reuse_times:
                    stats["entropy"] += entropy
                    stats["kl"] += kl
                    stats["clip_fraction"] += clip_fraction
        for k in stats.keys():
            stats[k] /= args.num_minibatch
        if args.wandb_show:
            from .logger import log
            log("train", T, stats)
        stats["T"] = T
        return stats

    def load(self, load_file):
        data = torch.load(load_file)
        self.policy.load_state_dict(data["policy"])
        self.optimizer.load_state_dict(data["optimizer"])
        return data["env_wrapper"]
    
    def save(self, env_wrapper, save_dir, file_name):
        data = {"env_wrapper": env_wrapper,
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict()}
        torch.save(data, os.path.join(save_dir, file_name))