import numpy as np

from .base import BaseBuffer


class NormalBuffer(BaseBuffer):
    """Normal Buffer directly stores all transitions
    
    All data are returned in np.float32/np.long format and [Batch_Size, Dim] shape
    so it can be used for pytorch process without any dtype transformation
    
    The data is stored on the CPU
    """

    def __init__(self, env):
        args = self.get_args()
        self.kicking = args.buffer_type
        self.S, self.L = args.buffer_capacity, 0
        self.discrete = env.discrete
        self.cur = 0 # The cursor locates at the idex of the next upcoming data
        
        # Now we make a new Transition-type and manually align the stride to a multiple of eight
        self.obs_shape = env.n_obs
        if isinstance(self.obs_shape, int):
            self.obs_shape = (self.obs_shape, )
        raw_transition = np.dtype([
            ("obs", np.float32, self.obs_shape),
            ("action", int) if self.discrete
                else ("action", np.float32, (env.n_act, )),
            ("reward", np.float32),
            ("done", bool)
        ])
        pad_byte = -(np.zeros(1, dtype=raw_transition).strides[0]) % 8
        self.pad_data = np.zeros(pad_byte, dtype=np.uint8)
        self.trans_dtype = np.dtype([
            ("obs", np.float32, self.obs_shape),
            ("action", int) if self.discrete
                else ("action", np.float32, (env.n_act, )),
            ("reward", np.float32),
            ("done", bool),
            ("pad", np.uint8, pad_byte)
        ])
        self.empty_data = (
            np.zeros(self.obs_shape, dtype=np.float32),
            0 if self.discrete else np.zeros(env.n_act, dtype=np.float32),
            0.,
            True,
            self.pad_data
        )
        self.stack = np.array([self.empty_data] * self.S, dtype=self.trans_dtype)
        self.next_obs_stack = None
        
    def add(self, obs, action, reward, done, next_obs=None):
        self.stack[self.cur] = (obs, action, reward, done, self.pad_data)
        if next_obs is not None:
            if self.next_obs_stack is None:
                self.next_obs_stack = np.zeros((self.S, *self.obs_shape), dtype=np.float32)
            self.next_obs_stack[self.cur] = next_obs
        if self.L < self.S:
            self.L += 1
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
        data = self.stack[idx]
        obs, action, reward = data["obs"], data["action"], data["reward"]
        done = data["done"].astype(np.float32)
        if collect_next_obs:
            next_obs = self.get_next_obs(idx)
            return {"obs": obs, "action": action, "reward": reward, "done": done, "next_obs": next_obs}
        else:
            return {"obs": obs, "action": action, "reward": reward, "done": done}
    
    def get_next_obs(self, idx):
        if self.next_obs_stack is None:
            return self.stack[idx + 1]
        else:
            return self.next_obs_stack[idx]
    
    def clearall(self):
        self.L, self.cur = 0, 0
        self.stack = np.array([self.empty_data] * self.S, dtype=self.trans_dtype)
    
    def __len__(self):
        return self.L