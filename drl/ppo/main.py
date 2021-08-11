import argparse
import os
from copy import deepcopy

import torch
import tqdm

from .agent import PPOAgent


def get_args():
    """PPO core hyper-parameters    
    ---------------------------------------------------------------------------
    TODO
    
    More parameters for record, test and other stuffs
    ---------------------------------------------------------------------------
    load_file    |   provide files storing pretrained models, or the training will be from scratch
    save_dir     |   help="the folder where models and training statictis will be saved
    render       |   render the enviroment during test time
    test         |   purely evaluate an agent without any training
    test_times   |   number of episodes to do a test
    test_interval         |   number of epochs between two tests
    checkpoint_interval   |   number of interaction steps to save a model backup when it > 0

    And there are some additional hyper-parameters specific for CNN architecture defined in
    the argument_complement function
    ---------------------------------------------------------------------------
    channel_divider   |   reduce channels by a divider in each layer from top to bottom
    kernel_size       |   a list of kernel sizes in each layer from top to bottom
    stride            |   a list of stride values in each layer from top to bottom
    """
    parser = argparse.ArgumentParser(description="ppo parser")
    parser.add_argument("--policy_lr", default=3e-4, type=float, \
        help="the learning rate of the policy network")
    parser.add_argument("--critic_lr", default=1e-3, type=float, \
        help="the learning rate of the critic network")
    parser.add_argument("--gamma", default=0.99, type=float, \
        help="discounting factor γ for future reward")
    parser.add_argument("--lam", default=0.95, type=float, \
        help="discounting factor λ for generalized advantage estimate")
    parser.add_argument("--clip_ration", default=0.2, type=float, \
        help="ppo clipping parameter")
    parser.add_argument("--entropy_bonus", default=0, type=float, \
        help="coefficient for the entropy term in the objective function")
    parser.add_argument("--target_kl", default=0.01, type=float, \
        help="early stoping if KL divergence between old/new polices exceeds a threshold")
    parser.add_argument("--hidden_dim", default=256, type=int, \
        help="number of hidden nodes per mlp layer/channels per cnn layer")
    parser.add_argument("--num_epoch", default=500, type=int, \
        help="number of ppo epochs to train an agent")
    parser.add_argument("--sample_steps", default=2048, type=int, \
        help="number of environment-interacting (sampling) steps in one ppo epoch")
    parser.add_argument("--reuse_times", default=4, type=int, \
        help="reuse the sampled data for multiple times for data efficiency")
    parser.add_argument("--num_minibatch", default=1, type=int, \
        help="split the whole data into a few minibatches during learning")

    # Other necessary parameters for a complete experiment
    parser.add_argument("--load_file", default=None, type=str, \
        help="provide files storing pretrained models, or the training will be from scratch")
    parser.add_argument("--save_dir", default="/results/", type=str, \
        help="the folder where models and training statictis will be saved")
    parser.add_argument("--render", default=False, action="store_true", \
        help="render the enviroment during test time")
    parser.add_argument("--test", default=False, action="store_true", \
        help="purely evaluate an agent without any training")
    parser.add_argument("--test_times", default=10, type=int, \
        help="number of episodes to do a test")
    parser.add_argument("--test_interval", default=10, type=int, \
        help="number of epochs between two tests")
    parser.add_argument("--checkpoint_interval", default=-1, type=int, \
        help="number of interaction steps to save a model backup when it > 0")

    argument_complement(parser)
    return parser.parse_args()

def test(args, agent, env):
    score = 0
    state = env.reset()
    for _ in range(args.max_episode_length):
        action = agent.act(state, deterministic=True)
        state, reward, done, info = env.step(action)
        if args.render: env.render()
        score += reward
        if done: break
    return score

def run(env, device, buffer):
    args = get_args()
    if args.algo != "ppo":
        raise ValueError("unexpected envoking with algorithm not set to be ppo")
    
    test_env = deepcopy(env)
    test_env.eval()
    agent = PPOAgent(args, env, device)
    if args.load_file is not None:
        agent.load(os.path.join(os.getcwd(), args.load_file))
    save_dir = os.path.join(os.getcwd(), args.save_dir)
    if args.test:
        avg_score = 0
        for _ in range(args.test_times):
            score = test(args, agent, test_env)
            print("episode #{}: score = {:.2f}".format(_, score))
            avg_score += score
        avg_score /= args.test_times
        print("")
        print("avg score = {:.2f}".format(avg_score))
        return

    tqdm_bar = tqdm.tqdm(range(1, 1 + args.num_epoch * args.sample_steps))
    best_avg_score = None
    stats = {}
    total_step = 0
    for epoch in range(1, args.num_epoch):
        buffer.clearall()
        obs = env.reset()
        for _ in range(args.sample_steps):
            total_step += 1
            tqdm_bar.update(1)
            if args.checkpoint_interval > 0 and total_step % args.checkpoint_interval == 0:
                agent.save(save_dir, "checkpoint_{}".format(total_step))
        tqdm_bar.set_description("Epoch #{}: ".format(epoch))
        if epoch % args.test_interval == 0:
            avg_score = 0
            for _ in range(args.test_times):
                score = test(args, agent, test_env)
                avg_score += score
            avg_score /= args.test_times
            if best_avg_score is None or avg_score > best_avg_score:
                best_avg_score = avg_score
                agent.save(save_dir, "best")
            print("---------- current score / best score = {:.2f} / {:.2f}".format(avg_score, best_avg_score))
        torch.save(stats, os.join(save_dir, "stats.pt"))

def argument_complement(parser):
    """The purpose of this is to complete the argument, so that we can use a complete parser 
    to check if there is any typo in the command line
    """
    # Base configuration
    parser.add_argument("--exp_name", default="unnamed", type=str, \
        help="the name of the experiment; the result will be saved at results/exp_name by default")
    parser.add_argument("--seed", default=0, type=int, \
        help="random seed of the whole experiment under which the result should be the same")
    parser.add_argument("--env_type", default="control", choices=["control", "atari", "mujoco"], \
        help="the type of the environment, with relative code and annotation at envs/env_type.py")
    parser.add_argument("--env", default="CartPole-v1", type=str, help="environment to interact with")
    parser.add_argument("--algo", default="dqn", \
        choices=["dqn", "vpg", "ddpg", "rainbow", "ppo", "td3", "sac"], \
        help="deep learning algorithm to choose")
    
    # Atari ALE configuration
    parser.add_argument("--screen_size", type=int, default=84, \
        help="clip the image into L*L squares")
    parser.add_argument("--sliding_window", type=int, default=4, \
        help="keep a series of contiguous frames as one observation (represented as a S*L*L tensor)")
    parser.add_argument("--max_episode_length", type=int, default=10000, \
        help="the maximum steps in one episode to enforce an early stop in some case")
    
    # Buffer configuration
    parser.add_argument("--buffer_type", default="dequeue", choices=["dequeue", "reservoir"], \
        help="the way to kick out old data when the buffer is full")
    parser.add_argument("--buffer_capacity", default=4096, type=int, \
        help="the maximum number of trainsitions the buffer can hold")
    
    # CNN architecture hyper-parameters
    parser.add_argument("--channel_divider", default="2,1,1", type=str, \
        help="reduce channels by a divider in each layer from top to bottom")
    parser.add_argument("--kernel_size", default="8,4,3", type=str, \
        help="a list of kernel sizes in each layer from top to bottom")
    parser.add_argument("--stride", default="4,2,1", type=str, \
        help="a list of stride values in each layer from top to bottom")
