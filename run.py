import argparse

import torch
import numpy as np

import buffers
import drl
import envs


def get_args():
    """Basic configuration
    ---------------------------------------------------------------------------
    exp_name       |   the name of the experiment; the result will be saved at results/exp_name by default
    seed           |   random seed of the whole experiment under which the result should be the same
    scenario       |   the type/background of the environment
    env            |   environment to interact with, here we give some typical example
                       control: CartPole-v1, Acrobot-v0, MountainCarContinuous-v0
                       atari: pong, qbert, freeway
                       mujuco: HalfCheetah, Ant, Humanoid
    num_env        |   number of parallel environments
    algo           |   deep learning algorithm to choose
    disable_cuda   |   cpu training even when gpus are available
    """
    parser = argparse.ArgumentParser(description="base configuration parser")
    parser.add_argument("--exp_name", default="unnamed", type=str, \
        help="the name of the experiment; the result will be saved at results/exp_name by default")
    parser.add_argument("--seed", default=0, type=int, \
        help="random seed of the whole experiment under which the result should be the same")
    parser.add_argument("--scenario", default="control", choices=["control", "atari", "mujoco"], \
        help="the type/background of the environment")
    parser.add_argument("--env", default="CartPole-v1", type=str, \
        help="environment to interact with")
    parser.add_argument("--num_env", default=1, type=int, \
        help="number of parallel environments")
    parser.add_argument("--algo", default="dqn", \
        choices=["dqn", "a2c", "ddpg", "rainbow", "ppo", "td3", "sac"], \
        help="deep learning algorithm to choose")
    parser.add_argument("--disable_cuda", default=False, action="store_true", \
        help="cpu training even when gpus are available")
    return parser.parse_known_args()[0]

args = get_args()
print("\n------ basic experimental configuration ------")
for key, val in vars(args).items():
    print("{:>20}   |   {}".format(key, val))
print("----------------------------------------------\n")

np.random.seed(args.seed)
torch.manual_seed(np.random.randint(1e9))
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True # See https://pytorch.org/docs/stable/notes/randomness.html
if torch.cuda.is_available() and not args.disable_cuda:
    device = torch.device("cuda")
    torch.cuda.manual_seed(np.random.randint(1e9))
else:
    device = torch.device("cpu")
torch.set_num_threads(4)

env, test_env = envs.make_env(args)
test_env.eval()
buffer = buffers.make_buffer(args, env)

if args.algo == "dqn":
    raise NotImplementedError
elif args.algo == "a2c":
    drl.A2C(env, test_env, device, buffer)
elif args.algo == "ppo":
    drl.PPO(env, test_env, device, buffer)
elif args.algo == "td3":
    drl.TD3(env, test_env, device, buffer)
else:
    raise ValueError("algorithm not defined")