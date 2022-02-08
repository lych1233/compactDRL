import numpy as np
import torch

from .base import BaseBuffer


class ImageBuffer(BaseBuffer):
    """For the Image Buffer, it stores all transitions in the following way for the sake of memory reduction:

    An observation is a tensor of [Frames, W, H] in [0, 1). We only stores the last frame in uint8 format.
    Typically for O \in [F, W, H], we store int(O[-1, :W, :H] * 255)
    
    All other data are in np.float32/np.long form so it can be used for pytorch processing without a dtype transformation
    The data will be stored on the CPU
    """

    def __init__(self, env, device):
        args = self.get_args()
        self.device = device
        self.kicking = args.buffer_type
        self.S, self.L = args.buffer_capacity, 0
        self.discrete = env.discrete
        self.cur = 0 # The cursor locates at the idex of the next upcoming data
        
        # First make a stack of frames
        self.frames_per_obs, self.frame_shape = env.n_obs[0], env.n_obs[1:]
        self.frames = np.zeros((self.S, *self.frame_shape), dtype=np.uint8)
        
        # Now we make a new Transition-type that does not contain the observation and manually align the stride to a multiple of eight
        raw_transition = np.dtype([
            ("action", int) if self.discrete
                else ("action", np.float32, (env.n_act, )),
            ("reward", np.float32),
            ("done", bool)
        ])
        pad_byte = -(np.zeros(1, dtype=raw_transition).strides[0]) % 8
        self.pad_data = np.zeros(pad_byte, dtype=np.uint8)
        self.trans_dtype = np.dtype([
            ("action", int) if self.discrete
                else ("action", np.float32, (env.n_act, )),
            ("reward", np.float32),
            ("done", bool),
            ("pad", np.uint8, pad_byte)
        ])
        self.empty_data = (
            0 if self.discrete else np.zeros(env.n_act, dtype=np.float32),
            0.,
            True,
            self.pad_data
        )
        self.stack = np.array([self.empty_data] * self.S, dtype=self.trans_dtype)
        self.next_frame = None
        
    def add(self, obs, action, reward, done, next_obs=None):
        if self.L < self.S:
            self.L += 1
        self.frames[self.cur] = obs[-1] * 256
        if next_obs is not None:
            self.next_frame = next_obs[-1]
        self.stack[self.cur] = (action, reward, done, self.pad_data)
        if self.L < self.S:
            self.cur += 1
        else:
            if self.kicking == "dequeue":
                self.cur = (self.cur + 1) % self.S
            elif self.kicking == "reservoir":
                raise ValueError("The image buffer only supports continuous storage, please use norml_buffer as a replacement")
            else:
                raise ValueError("The buffer type {} is not defined".format(self.kicking))
    
    def get(self, idx, collect_next_obs=False, terms=("obs", "action", "reward", "done")):
        if self.L < self.S and np.max(idx) >= self.L:
            raise ValueError("The index {} is larger than current buffer size {}".format(np.max(idx), self.L))
        idx %= self.S
        if collect_next_obs and "next_obs" not in terms:
            terms = (*terms, "next_obs")
        ret = {}
        if "obs" in terms:
            ret["obs"] = self.get_obs(idx)[0]
        data = self.stack[idx]
        if "action" in terms:
            ret["action"] = torch.as_tensor(data["action"]).to(self.device)
        if "reward" in terms:
            ret["reward"] = torch.as_tensor(data["reward"]).to(self.device)
        if "done" in terms:
            ret["done"] = torch.FloatTensor(data["done"]).to(self.device)
        if "next_obs" in terms:
            ret["next_obs"] = self.get_next_obs(idx)[0]
        return ret
    
    def get_obs(self, idx):
        obs_frames_idx = np.expand_dims(idx, -1) + np.arange(self.frames_per_obs) - self.frames_per_obs + 1
        obs = torch.FloatTensor(self.frames[obs_frames_idx]).to(self.device) / 256
        mask = np.copy(self.stack[obs_frames_idx]["done"])
        mask[(idx - (self.cur - 1)) % self.S == 0] = True
        mask_shape = mask.shape
        mask = mask.reshape(-1, mask_shape[-1])
        mask[:, -1] = False
        for i in range(mask.shape[-1] - 2, -1, -1):
            mask[:, i] = np.logical_or(mask[:, i], mask[:, i + 1])
        mask = mask.reshape(mask_shape)
        obs[mask] = 0
        return obs, mask
    
    def get_next_obs(self, idx):
        next_obs, mask = self.get_obs((idx + 1) % self.S)
        if self.next_frame is not None:
            next_obs[idx == self.cur] = torch.FloatTensor(self.next_frame).to(self.device)
            next_obs[mask] = 0
        return next_obs, mask
    
    def clearall(self):
        self.L, self.cur = 0, 0
        self.stack = np.array([self.empty_data] * self.S, dtype=self.trans_dtype)
    
    def __len__(self):
        return self.L