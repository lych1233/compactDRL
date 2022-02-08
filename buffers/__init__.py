from .normal_buffer import NormalBuffer
from .image_buffer import ImageBuffer
from .vector_buffer import VectorBuffer

def make_buffer(args, env, device):
    """Since we only activate environment parallelization for on-policy algorithms, we use vectorized buffer only for them.
    The data is stored on the CPU.
    """
    if args.algo in ["a2c", "ppo"]:
        return VectorBuffer(env, device, args.num_env)
    if args.scenario in ["control", "mujoco"]:
        return NormalBuffer(env, device)
    elif args.scenario in ["atari"]:
        return ImageBuffer(env, device)
    else:
        raise NotImplementedError