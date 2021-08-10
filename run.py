import argparse

import torch
import numpy as np

import buffers
import drl
import envs


def get_args():
    """Basic configuration
    ---------------------------------------------------------------------------
    exp_name   |   the name of the experiment; the result will be saved at results/exp_name by default
    seed       |   random seed of the whole experiment under which the result should be the same
    env_type   |   the type of the environment, with relative code and annotation at envs/env_type.py
    env        |   environment to interact with, here we give some typical example
                       control: CartPole-v1, Acrobot-v0, MountainCarContinuous-v0
                       atari: pong, qbert, freeway
                       mujuco: HalfCheetah, Ant, Humanoid
    algo       |   deep learning algorithm to choose
    """
    parser = argparse.ArgumentParser(description="base configuration parser")
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
    return parser.parse_known_args()[0]

args = get_args()
print(vars(args))

env = envs.make_env(args)
np.random.seed(args.seed)
torch.manual_seed(np.random.randint(1e9))
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed(np.random.randint(1e9))
else:
    device = torch.device("cpu")

buffer = buffers.make_buffer(args, env, device)

if args.algo == "dqn":
    raise NotImplementedError
elif args.algo == "vpg":
    raise NotImplementedError
elif args.algo == "ppo":
    drl.PPO(env, device, buffer)
else:
    raise ValueError("algorithm not defined")