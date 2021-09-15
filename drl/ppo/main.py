import os
from collections import defaultdict
import warnings

import numpy as np
import torch
import tqdm

from .agent import PPOAgent
from .config import get_args


def test(args, agent, env):
    score_list = []
    for _ in range(args.test_times):
        state = env.reset()
        score = 0
        for __ in range(args.max_episode_length):
            action = agent.act(np.expand_dims(state, 0), deterministic=True)
            state, reward, done, info = env.step(action)
            if args.render:
                env.render()
            score += reward
            if done:
                break
        score_list.append(score)
    return np.mean(score_list), score_list

def run(env, test_env, device, buffer):
    args = get_args()
    if args.sample_steps % args.num_env != 0:
        import warnings
        warnings.warn("{} transitions will be dispelled due to vectorized environment cutting off".format(args.sample_steps % args.num_env))
        args.sample_steps -= args.sample_steps % args.num_env
    if args.algo != "ppo":
        raise ValueError("unexpected envoking with algorithm not set to be ppo")
    
    test_env.eval()
    agent = PPOAgent(args, env, device)
    if args.load_file is not None:
        agent.load(os.path.join(os.getcwd(), args.load_file))
    
    if args.test:
        avg_score, score_list = test(args, agent, test_env)
        print("score in last ten episodes: {}".format(score_list[-10:]))
        print("avg score = {:.2f}".format(avg_score))
        return
    
    save_dir = os.path.join(os.getcwd(), args.save_dir, args.exp_name + "_seed_" + str(args.seed))
    os.makedirs(save_dir, exist_ok=True)
    stats = defaultdict(list)
    if args.wandb_show:
        from .logger import wandb_init
        wandb_init(args, save_dir)

    tqdm_bar = tqdm.tqdm(range(1, 1 + args.num_T), ncols=120)
    num_epoch = args.num_T // args.sample_steps + 1
    obs, cur_score, best_avg_score = env.reset(), np.zeros(args.num_env), -1e100
    T, episode_len = 0, np.zeros(args.num_env, dtype=int)
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
            cur_score += reward
            buffer.add(obs, action, reward, done)
            obs = next_obs
            if np.any(done):
                new_score, new_episode = cur_score[done].mean().item(), episode_len[done].mean().item()
                for env_score in cur_score[done]:
                    stats["all_score"].append(env_score)
                stats["T_score_pair"].append((T, new_score))
                stats["T_episode_len_pair"].append((T, new_episode))
                if args.wandb_show:
                    from .logger import log
                    log("env", T, {"training score": new_score})
                    log("env", T, {"episode length": new_episode})
                cur_score[done] = 0
                episode_len[done] = 0
            
            if T % args.test_interval == 0:
                avg_score, score_list = test(args, agent, test_env)
                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    agent.save(save_dir, "best.pt")
                print("\n----- current score / best score = {:.2f} / {:.2f} -----\n".format(avg_score, best_avg_score))
                print(stats["learn_stats"][-1])
                print("")
                if args.wandb_show:
                    from .logger import log
                    log("env", T, {"testing score": avg_score})
            if T % args.checkpoint_interval == 0 and args.checkpoint_interval > 0:
                agent.save(save_dir, "checkpoint_{}.pt".format(T))
        
        learn_stats = agent.learn(args, buffer, obs, T)
        stats["learn_stats"].append(learn_stats)
        
        tqdm_bar.set_description(
            "Epoch #{}, T #{} | Rolling: {:.2f}".format(
            epoch, T, np.mean(stats["all_score"][-20:])))
        torch.save(stats, os.path.join(save_dir, "stats.pt"))
    
    if args.wandb_show:
        from .logger import wandb_finish
        wandb_finish()
