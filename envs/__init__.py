import numpy as np

from .control import ClassicControl
from .atari import AtariALE
from .mujoco import MuJoCoRobot


def make_env(args):
    if args.env_type == "control":
        fn = ClassicControl
    elif args.env_type == "atari":
        fn = AtariALE
    elif args.env_type == "mujoco":
        fn = MuJoCoRobot
    else:
        raise TypeError("env not defined")
    if args.algo in ["vpg", "ppo"]:
        base_seed = np.random.randint(1e9)
        from functools import partial
        fn_envs = [(partial(fn, args.env, base_seed + k)) for k in range(args.num_env)]
        from .vec_env import SubprocVecEnv, DummyVecEnv
        vec_env_wrapper = SubprocVecEnv if args.env_type == "atari" else DummyVecEnv
        return vec_env_wrapper(fn_envs), fn(args.env, np.random.randint(1e9))
    else:
        return fn(args.env, np.random.randint(1e9)), fn(args.env, np.random.randint(1e9))
