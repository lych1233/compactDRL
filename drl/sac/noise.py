from abc import ABC

import numpy as np


class OUNoise(ABC):
    """Based on https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    """
    def __init__(self, n_act, seed, theta=0.15, dt=0.01):
        super(OUNoise, self).__init__()
        self.n_act = n_act
        self.rng = np.random.RandomState(seed)
        self.noise = np.zeros(self.n_act)
        self.theta = theta
        self.dt = dt

    def __call__(self):
        d_noise = -self.theta * self.noise * self.dt + np.sqrt(self.dt) * np.random.randn(self.n_act)
        self.noise += d_noise
        return self.noise.astype(np.float32)
