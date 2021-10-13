import numpy as np


def make_env(args):
    """
    Rules for environment chosen:
        For on-policy algorithm, use vectorized environment
        For control/mujoco tasks, use observation normalization and reward (value) normalization in training for better performance
    Only activate environment parallelization and observation/reward normalization for on-policy algorithms like a2c and ppo. Therefore we implement observation normalization inside the vec_env.py.
    """
    if args.scenario == "control":
        from .control import ClassicControl
        fn = ClassicControl
    elif args.scenario == "atari":
        from .atari import AtariALE
        fn = AtariALE
    elif args.scenario == "mujoco":
        from .mujoco import MuJoCoRobot
        fn = MuJoCoRobot
    else:
        raise TypeError("env not defined")
    if args.algo in ["a2c", "ppo"]:
        base_seed = np.random.randint(1e9)
        from functools import partial
        fn_envs = [(partial(fn, args.env, base_seed + k)) for k in range(args.num_env)]
        from .vec_env import SubprocVecEnv, DummyVecEnv
        vec_env_wrapper = SubprocVecEnv if args.scenario in ["atari"] else DummyVecEnv
        return vec_env_wrapper(fn_envs), fn(args.env, np.random.randint(1e9))
    else:
        import warnings
        if args.num_env != 1:
            warnings.warn("For off-policy algorithms we will always use a single environment, even though {} environments are demanded".format(args.num_env))
        return fn(args.env, np.random.randint(1e9)), fn(args.env, np.random.randint(1e9))
