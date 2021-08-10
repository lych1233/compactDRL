import argparse

import torch

from .agent import PPOAgent


def get_args():
    """PPO hyper-parameters
    
    ---------------------------------------------------------------------------
    policy_lr   |   learning rate for the policy network
    critic_lr   |   learning rate for the critic network
    batch_size   |   batch size in one neural network learing step
    """
    parser = argparse.ArgumentParser(description="ppo parser")
    parser.add_argument("--policy-lr", default=3e-4, type=float, \
        help="the learning rate of deep q-network parameters")
    parser.add_argument("--critic-lr", default=1e-3, type=float, \
        help="learning rate for the critic network")
    parser.add_argument("--batch-size", default=2048, type=int, \
        help="batch size in one neural network learing step")
    argument_complement(parser)
    return parser.parse_args()

def run(env, device, buffer):
    args = get_args()
    if args.algo != "ppo":
        raise ValueError("unexpected envoking with algorithm not set to be ppo")
    for key, value in vars(args).items():
        print("key = {}, value = {}".format(key, value))
    print("cuda ok? {}, deivce =  {}".format(torch.cuda.is_available(), device))
    print("buffer length = {}".format(len(buffer)))
    agent = PPOAgent(args)

    print("dqn part run successfully!")


def argument_complement(parser):
    """The purpose of this is to complete the argument, so that we can use a complete parser 
    to check if there is any typo in the command line
    """
    # The base configuration
    parser.add_argument("--exp-name", default="unnamed", type=str, \
        help="the name of the experiment; the result will be saved at results/exp_name by default")
    parser.add_argument("--seed", default=0, type=int, \
        help="random seed of the whole experiment under which the result should be the same")
    parser.add_argument("--env-type", default="control", choices=["control", "atari", "mujoco"], \
        help="the type of the environment, with relative code and annotation at envs/env_type.py")
    parser.add_argument("--env", default="CartPole-v1", type=str, help="environment to interact with")
    parser.add_argument("--algo", default="dqn", \
        choices=["dqn", "vpg", "ddpg", "rainbow", "ppo", "td3", "sac"], \
        help="deep learning algorithm to choose")
    
    # Atari ALE configuration
    parser.add_argument("--screen-size", type=int, default=84, \
        help="clip the image into L*L squares")
    parser.add_argument("--sliding-window", type=int, default=4, \
        help="keep a series of contiguous frames as one observation (represented as a S*L*L tensor)")
    parser.add_argument("--max-episode-length", type=int, default=10000, \
        help="the maximum steps in one episode to enforce an early stop in some case")
    
    # Buffer configuration
    parser.add_argument("--buffer-type", default="dequeue", choices=["dequeue", "reservoir"])
    parser.add_argument("--buffer-capacity", default=4096, type=int, \
        help="the maximum number of trainsitions the buffer can hold")