from envs.base import BaseEnv
from envs.control import Control
from envs.atari import AtariALE

def make_env(args):
    if args.env_type == "control":
        return Control(args.env)
    elif args.env_type == "atari":
        return AtariALE(args.env)
    elif args.env_type == "mujoco":
        raise NotImplementedError
    else:
        raise TypeError("env not defined")
