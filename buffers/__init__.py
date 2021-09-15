from .normal_buffer import NormalBuffer
from .image_buffer import ImageBuffer
from .vector_buffer import VectorBuffer

def make_buffer(args, env, device):
    if args.algo in ["vpg", "ppo"]:
        return VectorBuffer(env, args.num_env)
    if args.env_type in ["control", "mujuco"]:
        return NormalBuffer(env)
    elif args.env_type in ["atari"]:
        return ImageBuffer(env)
    else:
        raise NotImplementedError