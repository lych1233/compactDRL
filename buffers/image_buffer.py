import numpy as np

from .base import BaseBuffer


class ImageBuffer(BaseBuffer):
    """For the Image Buffer, it stores all transitions in the following way for the sake of memory reduction:

    An observation is a tensor of [Frames, W, H] in [0, 1). We only stores the last frame in uint8 format.
    Typically for O \in [F, W, H], we store int(O[-1, :W, :H] * 255)
    
    All other data are in np.float32/np.long form so it can be used for pytorch processing without a dtype transformation
    The data will be stored on the CPU
    """

    def __init__(self, env):
        args = self.get_args()
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
        
    def add(self, obs, action, reward, done, next_obs=None):
        if self.L < self.S:
            self.L += 1
        self.frames[self.cur] = obs[-1] * 256
        if next_obs is not None:
            self.frames[(self.cur + 1) % self.S] = next_obs[-1] * 256
        self.stack[self.cur] = (action, reward, done, self.pad_data)
        if self.L < self.S:
            self.cur += 1
        else:
            if self.kicking == "dequeue":
                self.cur = (self.cur + 1) % self.S
            elif self.kicking == "reservoir":
                self.cur = np.random.randint(self.S)
            else:
                raise ValueError("The buffer type {} is not defined".format(self.kicking))
    
    def get(self, idx, collect_next_obs=False):
        if np.max(idx) >= self.L:
            raise ValueError("The index {} is larger than current buffer size {}".format(np.max(idx), self.L))
        obs = self.get_obs(idx)
        data = self.stack[idx]
        action, reward = data["action"], data["reward"]
        done = data["done"].astype(np.float32)
        if collect_next_obs:
            next_obs = self.get_obs(idx + 1)
            return {"obs": obs, "action": action, "reward": reward, "done": done, "next_obs": next_obs}
        else:
            return {"obs": obs, "action": action, "reward": reward, "done": done}
    
    def get_obs(self, idx):
        obs_frames_idx = np.expand_dims(idx, 1) + np.arange(self.frames_per_obs) - self.frames_per_obs + 1
        obs_frames_idx %= self.S
        obs = (self.frames[obs_frames_idx] / 256).astype(np.float32)
        mask = np.copy(self.stack[obs_frames_idx]["done"])
        for i in range(mask.shape[-1] - 2, -1, -1):
            mask[:, i] = np.logical_or(mask[:, i], mask[:, i + 1])
        obs[mask] = 0
        return obs
    
    def clearall(self):
        self.L, self.cur = 0, 0
        self.stack = np.array([self.empty_data] * self.S, dtype=self.trans_dtype)
    
    def __len__(self):
        return self.L