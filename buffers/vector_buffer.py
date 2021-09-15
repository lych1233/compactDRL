import numpy as np

from .base import BaseBuffer


class VectorBuffer(BaseBuffer):
    """Vectorized Buffer directly stores all transitions from all environments
    
    All data are returned in np.float32/np.long format and [Batch_Size, Num_Env, Dim] shape
    so it can be used for pytorch process without any dtype transformation
    
    The data is stored on the CPU
    """

    def __init__(self, env, num_env):
        args = self.get_args()
        self.kicking = args.buffer_type
        self.n, self.S, self.L = num_env, args.buffer_capacity // num_env, 0
        print(self.n, self.S)
        self.discrete = env.discrete
        self.cur = 0 # The cursor locates at the idex of the next upcoming data
        
        # Now we make a new Transition-type and manually align the stride to a multiple of eight
        self.obs_shape = env.n_obs
        if isinstance(self.obs_shape, int):
            self.obs_shape = (self.obs_shape, )
        raw_transition = np.dtype([
            ("obs", np.float32, (self.n, *self.obs_shape)),
            ("action", int, (self.n, )) if self.discrete
                else ("action", np.float32, (self.n, env.n_act)),
            ("reward", np.float32, (self.n, )),
            ("done", bool, (self.n, ))
        ])
        pad_byte = -(np.zeros(1, dtype=raw_transition).strides[0]) % 8
        self.pad_data = np.zeros(pad_byte, dtype=np.uint8)
        self.trans_dtype = np.dtype([
            ("obs", np.float32, (self.n, *self.obs_shape)),
            ("action", int, (self.n, )) if self.discrete
                else ("action", np.float32, (self.n, env.n_act)),
            ("reward", np.float32, (self.n, )),
            ("done", bool, (self.n, )),
            ("pad", np.uint8, pad_byte)
        ])
        self.empty_data = (
            np.zeros((self.n, *self.obs_shape), dtype=np.float32),
            np.zeros(self.n, dtype=int) if self.discrete
                else np.zeros((self.n, env.n_act), dtype=np.float32),
            np.zeros(self.n, dtype=np.float32),
            np.ones(self.n, dtype=bool),
            self.pad_data
        )
        self.stack = np.array([self.empty_data] * self.S, dtype=self.trans_dtype)
        self.next_obs_stack = None
        
    def add(self, obs, action, reward, done, next_obs=None):
        self.stack[self.cur] = (obs, action, reward, done, self.pad_data)
        if next_obs is not None:
            if self.next_obs_stack is None:
                self.next_obs_stack = np.zeros((self.S, self.n, *self.obs_shape), dtype=np.float32)
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
    
    def get(self, idx):
        if np.max(idx) >= self.L:
            raise ValueError("The index {} is larger than current buffer size {}".format(np.max(idx), self.L))
        data = self.stack[idx]
        obs, action, reward = data["obs"], data["action"], data["reward"]
        done = data["done"].astype(np.float32)
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