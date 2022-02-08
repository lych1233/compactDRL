import os
from collections import defaultdict

import numpy as np
import torch
import tqdm

from .agent import RainbowAgent
from .prioritized_sampling import SamplingPool
from .config import get_args


def test(args, agent, env):
    score_list = []
    for _ in range(args.test_times):
        obs = env.reset()
        score = 0
        for __ in range(args.max_episode_length):
            action = agent.act(obs, eps=0)
            obs, reward, done, info = env.step(action)
            if args.render:
                env.render()
            score += reward
            if done:
                break
        score_list.append(score)
    return np.mean(score_list), score_list

def run(env, test_env, device, buffer):
    args = get_args()
    assert args.algo == "rainbow", "unexpected envoking with algorithm not set to be rainbow"
    assert env.discrete, "rainbow only deals with continuous action space"
    import warnings
    if "noisy_net" in args.enhancement and args.eps > 0:
        warnings.warn("For noisy net exploration we do not adopt eps-greedy exploration anymore")
        args.eps = 0
    if "multi_step" not in args.enhancement and args.n_steps != 1:
        warnings.warn("One-step boostrap will be used when disabling multi_step, although n_steps={} is set".format(args.n_steps))
        args.n_steps = 1
    
    save_dir = os.path.join(os.getcwd(), args.save_dir, args.exp_name + "_seed_" + str(args.seed))
    os.makedirs(save_dir, exist_ok=True)

    test_env.eval()
    agent = RainbowAgent(args, env, device)
    pool = SamplingPool(args, buffer.S)
    if args.load_file is not None:
        agent.load(os.path.join(os.getcwd(), args.load_file))
    
    if args.test_model:
        avg_score, score_list = test(args, agent, test_env)
        print("score in last ten episodes: {}".format(score_list[-10:]))
        print("avg score = {:.2f}".format(avg_score))
        return
    
    avg_score, score_list = test(args, agent, test_env)
    best_avg_score = avg_score
    if args.wandb_show:
        from .logger import log, wandb_init
        wandb_init(args, save_dir)
        log("env", 0, {"testing score": avg_score})
    print("\n----- current score / best score = {:.2f} / {:.2f} -----\n".format(avg_score, best_avg_score))
    
    tqdm_bar = tqdm.tqdm(range(1, 1 + args.num_T), ncols=120)
    obs, T, episode, cur_score, episode_len = env.reset(), 0, 0, 0, 0
    stats = defaultdict(list)
    for T in tqdm_bar:
        agent.lr_decay(args, T, args.num_T)
        if T % args.update_frequency == 0:
            agent.reset_noise()

        episode_len += 1
        action = agent.act(obs, args.eps)
        next_obs, reward, done, info = env.step(action)
        cur_score += reward
        buffer.add(obs, action, reward, done)
        pool.add(T, args.num_T)
        if done:
            episode += 1
            stats["all_score"].append(cur_score)
            stats["train_T"].append(T)
            stats["train_score"].append(cur_score)
            stats["train_episode_len"].append(episode_len)
            if args.wandb_show:
                from .logger import log
                log("env", T, {"training score": cur_score})
                log("env", T, {"episode length": episode_len})
            cur_score, episode_len = 0, 0
            tqdm_bar.set_description(
                "Episode #{}, T #{} | Rolling: {:.2f}".format(
                episode, T, np.mean(stats["all_score"][-20:]))
            )
            obs = env.reset()
        else:
            obs = next_obs
        
        if T > args.start_learning and T % args.update_frequency == 0:
            learn_stats = agent.learn(args, buffer, pool, T)
            for k, v in learn_stats.items():
                stats["learn_" + k].append(v)
        
        if T % args.test_interval == 0:
            torch.save(stats, os.path.join(save_dir, "stats.pt"))
            avg_score, score_list = test(args, agent, test_env)
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                agent.save(save_dir, "best.pt")
            print("\n\n----- current score / best score = {:.2f} / {:.2f} -----\n".format(avg_score, best_avg_score))
            stats["test_T"].append(T)
            stats["test_score"].append(avg_score)
            if args.wandb_show:
                from .logger import log
                log("env", T, {"testing score": avg_score})
        if T % args.checkpoint_interval == 0 and args.checkpoint_interval > 0:
            agent.save(save_dir, "checkpoint_{}.pt".format(T))
    
    if args.wandb_show:
        from .logger import wandb_finish
        wandb_finish()
