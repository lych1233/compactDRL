from buffers.base import BaseBuffer
from buffers.normal_buffer import NormalBuffer
from buffers.image_buffer import ImageBuffer

def make_buffer(args, env, device):
    if args.env_type in ["control", "mujuco"]:
        return NormalBuffer(env)
    elif args.env_type in ["atari"]:
        return ImageBuffer(env)
    else:
        raise NotImplementedError