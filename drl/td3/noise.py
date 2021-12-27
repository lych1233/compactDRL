from abc import ABC

import numpy as np


class Noise(ABC):
    def __init__(self, n_act, seed):
        super(Noise, self).__init__()
        self.n_act = n_act
        self.rng = np.random.RandomState(seed)

class GaussianNoise(Noise):
    def __init__(self, n_act, seed):
        super(GaussianNoise, self).__init__(n_act, seed)

    def __call__(self):
        return self.rng.randn(self.n_act).astype(np.float32)

class OUNoise(Noise):
    """Based on https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
    """
    def __init__(self, n_act, seed, theta=0.15, dt=0.01):
        super(OUNoise, self).__init__(n_act, seed)
        self.noise = np.zeros(self.n_act)
        self.theta = theta
        self.dt = dt

    def __call__(self):
        d_noise = -self.theta * self.noise * self.dt + np.sqrt(self.dt) * np.random.randn(self.n_act)
        self.noise += d_noise
        return self.noise.astype(np.float32)
