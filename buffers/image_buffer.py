# TODO test it after the normal_buffer
import argparse

import numpy as np

from buffers.base import BaseBuffer


class ImageBuffer(BaseBuffer):
    """For the Image Buffer:
    An observation is a tensor of L * W * H in [0, 1). We only stores the first frame by uint8.
    Typically for O[L, W, H], we store int(O[0, W, H] * 255)
    
    All data are in np.float32/np.long form so it can be used for torch without a dtype transformation
    The data will be stored on the CPU
    """

    def __init__(self, env):
        raise NotImplementedError
        args = self.get_args()
        self.kicking = args.kicking
        self.S, self.L = args.buffer_capaciy, 0
        self.cur = 0 # The cursor locates at the idex of the next upcoming data
        
        # Now we make a new Transition-type and manually align the stride to a multiple of eight
        frame_shape = env.n_obs[1:]
        raw_transition = np.dtype([("obs", (np.uint8, frame_shape)),
                                   ("action", np.long if env.discrete else (np.float32, env.n_act)),
                                   ("reward", np.float32),
                                   ("done", np.bool_)])
        pad_byte = -(np.zeros(1, raw_transition).strides[0]) % 8
        self.pad_data = np.zeros(pad_byte, dtype=np.uint8)
        self.trans_dtype = np.dtype([("obs", (np.uint8, frame_shape)),
                                     ("action", np.long if env.discrete else (np.float32, env.n_act)),
                                     ("reward", np.float32),
                                     ("done", np.bool_),
                                     ("pad", np.uint8, pad_byte)])
        if env.discrete:
            self.empty_data = (np.zeros(env.n_obs, dtype=np.float32), 0, 0., True, self.pad_data)
        else:
            self.empty_data = (np.zeros(env.n_obs, dtype=np.float32), np.zeros(env.n_act, dtype=np.float32), 0., True, self.pad_data)
        self.stack = np.array([self.empty_data] * self.S, dtype=self.trans_dtype)
        
    def add(self, obs, action, reward, done, next_obs=None):
        self.stack[self.cur] = (obs, action, reward, done, self.pad_data)
        if self.L < self.S:
            self.L += 1
        else:
            if self.kicking == "dequeue":
                self.cur = (self.cur + 1) % self.S
            elif self.kicking == "reservoir":
                self.cur = np.random.randint(self.S)
            else:
                raise ValueError("The buffer type {} is not defined".format(self.kicking))
    
    def get(self, idx):
        if np.max(idx) >= self.L:
            raise ValueError("The index {} is larger than current buffer size {}".format(np.max(idxs), self.L))
        data = self.stack[idx]
        obs, action, reward, done = data["obs"], data["action"], data["reward"], data["done"]
        next_obs = self.stack[(idx + 1) % self.S]["obs"]
        return obs, action, reward, done, next_obs
    
    def clearall(self):
        self.L, self.cur = 0, 0
        self.stack = np.array([self.empty_data] * self.S, dtype=self.trans_dtype)
    
    def __len__(self):
        return self.L