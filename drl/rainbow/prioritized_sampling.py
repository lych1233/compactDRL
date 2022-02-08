import numpy as np


class SegmentTree(object):
    def __init__(self, n):
        """The structure is like 1 -> (2, 3), 2 -> (4, 5) and so on, and it would be easy to find parents (x/2) and children (x*2, x*2+1)
        the original data p_i is stored on i + pow2, i.e., all in the deepest layer
        """
        self.n = n
        self.pow2 = 1 << int(np.log2(self.n) + 1)
        self.sum = np.zeros(self.pow2 + self.n, dtype=np.float64)
        
    def update(self, idx, value):
        idx += self.pow2
        self.sum[idx] = value
        while idx[0] > 1:
            idx >>= 1
            self.sum[idx] = self.sum[idx << 1] + self.sum[idx << 1 | 1]
    
    def find(self, batch_size, value):
        value *= self.sum[1]
        idx = np.ones(batch_size, dtype=int)
        while idx[0] < self.pow2:
            go_right = value > self.sum[idx << 1]
            value -= go_right * self.sum[idx << 1]
            idx = idx << 1 | go_right
        return idx - self.pow2
    
    def probs(self, idxs):
        return self.sum[idxs + self.pow2] / self.sum[1]


class SamplingPool(object):
    def __init__(self, args, S):
        self.S, self.L, self.cur, self.total_items = S, 0, 0, 0
        self.n_steps = args.n_steps
        
        self.prioritized_replay = "prioritized_replay" in args.enhancement
        if self.prioritized_replay:
            self.alpha = args.priority_alpha
            self.beta_0 = args.priority_beta
            self.max_P = 1
            self.sum_tree = SegmentTree(self.S)
    
    def add(self, cur, total):
        self.total_items += 1
        if self.L < self.S:
            self.L += 1
        if self.prioritized_replay:
            self.beta = self.beta_0 + (1 - self.beta_0) * (cur / total)
            self.update_priority(np.array([self.cur]), np.array([self.max_P]))
        self.cur = (self.cur + 1) % self.S
    
    def update_priority(self, idx, P):
        P = P.astype(np.float64)
        self.max_P = max(self.max_P, P.max())
        P = np.power(P, self.alpha)
        self.sum_tree.update(idx, P)
    
    def sample(self, batch_size):
        while True:
            if self.prioritized_replay:
                values = np.random.rand(batch_size)
                idx = self.sum_tree.find(batch_size, values)
            else:
                idx = np.random.choice(self.L, batch_size, replace=True)
            if np.all((self.cur - idx) % self.S > self.n_steps):
                break
        if self.prioritized_replay:
            sampling_prob = self.sum_tree.probs(idx)
            importance_factor = (sampling_prob * self.L) ** -self.beta
        else:
            importance_factor = np.ones(batch_size)
        return idx, importance_factor

    def __len__(self):
        return self.L
