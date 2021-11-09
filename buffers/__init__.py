from .normal_buffer import NormalBuffer
from .image_buffer import ImageBuffer
from .vector_buffer import VectorBuffer

def make_buffer(args, env):
    """Since we only activate environment parallelization for on-policy algorithms, we use vectorized buffer only for them.
    The data is stored on the CPU.
    """
    if args.algo in ["a2c", "ppo"]:
        return VectorBuffer(env, args.num_env)
    if args.scenario in ["control", "mujoco"]:
        return NormalBuffer(env)
    elif args.scenario in ["atari"]:
        return ImageBuffer(env)
    else:
        raise NotImplementedError