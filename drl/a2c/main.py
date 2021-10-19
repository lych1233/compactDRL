import os
import warnings
from collections import defaultdict

import numpy as np
import torch
import tqdm

from .agent import A2CAgent
from .config import get_args


def test(args, agent, env, obs_normalizer):
    score_list = []
    for _ in range(args.test_times):
        obs = env.reset()
        score = 0
        for __ in range(args.max_episode_length):
            if obs_normalizer:
                obs = obs_normalizer.normalize(obs)
            action = agent.act(np.expand_dims(obs, 0), deterministic=True)
            obs, reward, done, info = env.step(np.squeeze(action, 0))
            if args.render:
                env.render()
            score += reward
            if done:
                break
        score_list.append(score)
    return np.mean(score_list), score_list

def run(env, test_env, device, buffer):
    args = get_args()
    if args.algo != "a2c":
        raise ValueError("unexpected envoking with algorithm not set to be a2c")
    if args.sample_steps % args.num_env != 0:
        warnings.warn("{} transitions will be truancated due to vectorized environment cutting off".format(args.sample_steps % args.num_env))
        args.sample_steps -= args.sample_steps % args.num_env
    if args.test_interval % args.num_env != 0:
        warnings.warn("test_interval will be reduced by {}".format(args.test_interval % args.num_env))
        args.test_interval -= args.test_interval % args.num_env
    
    save_dir = os.path.join(os.getcwd(), args.save_dir, args.exp_name + "_seed_" + str(args.seed))
    os.makedirs(save_dir, exist_ok=True)

    test_env.eval()
    agent = A2CAgent(args, env, device)
    if args.load_file is not None:
        env = agent.load(os.path.join(os.getcwd(), args.load_file))
    
    if args.test_model:
        avg_score, score_list = test(args, agent, test_env, env.obs_normalizer)
        print("score in last ten episodes: {}".format(score_list[-10:]))
        print("avg score = {:.2f}".format(avg_score))
        return
    
    avg_score, score_list = test(args, agent, test_env, env.obs_normalizer)
    best_avg_score = avg_score
    if args.wandb_show:
        from .logger import log, wandb_init
        wandb_init(args, save_dir)
        log("env", 0, {"testing score": avg_score})
    print("\n----- current score / best score = {:.2f} / {:.2f} -----\n".format(avg_score, best_avg_score))
    
    tqdm_bar = tqdm.tqdm(range(1, 1 + args.num_T), ncols=120)
    num_epoch = args.num_T // args.sample_steps + 1
    obs, T = env.reset(), 0
    cur_score, episode_len = np.zeros(args.num_env), np.zeros(args.num_env, dtype=int)
    stats = defaultdict(list)
    for epoch in range(1, 1 + num_epoch):
        buffer.clearall()
        agent.lr_decay(args, epoch - 1, num_epoch)
        for _ in range(args.sample_steps // args.num_env):
            if T < args.num_T:
                tqdm_bar.update(min(args.num_env, args.num_T - T))
            T += args.num_env
            
            episode_len += 1
            action = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            cur_score += reward * env.reward_scale_factor
            buffer.add(obs, action, reward, done)
            obs = next_obs
            if np.any(done):
                new_score, new_episode_len = cur_score[done].mean().item(), episode_len[done].mean().item()
                for env_score in cur_score[done]:
                    stats["all_score"].append(env_score)
                stats["train_T"].append(T)
                stats["train_score"].append(new_score)
                stats["train_episode_len"].append(new_episode_len)
                if args.wandb_show:
                    from .logger import log
                    log("env", T, {"training score": new_score})
                    log("env", T, {"episode length": new_episode_len})
                cur_score[done] = 0
                episode_len[done] = 0
            
            if T % args.test_interval == 0:
                avg_score, score_list = test(args, agent, test_env, env.obs_normalizer)
                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    agent.save(env, save_dir, "best.pt")
                print("\n\n----- current score / best score = {:.2f} / {:.2f} -----\n".format(avg_score, best_avg_score))
                stats["test_T"].append(T)
                stats["test_score"].append(avg_score)
                if args.wandb_show:
                    from .logger import log
                    log("env", T, {"testing score": avg_score})
            if T % args.checkpoint_interval == 0 and args.checkpoint_interval > 0:
                agent.save(env, save_dir, "checkpoint_{}.pt".format(T))
        
        learn_stats = agent.learn(args, buffer, obs, T)
        for k, v in learn_stats.items():
            stats["learn_" + k].append(v)
        
        tqdm_bar.set_description(
            "Epoch #{}, T #{} | Rolling: {:.2f}".format(
            epoch, T, np.mean(stats["all_score"][-20:]))
        )
        torch.save(stats, os.path.join(save_dir, "stats.pt"))
    
    if args.wandb_show:
        from .logger import wandb_finish
        wandb_finish()
