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
        idx = np.ones(batch_size, dtype=np.long)
        while idx[0] < self.pow2:
            go_right = value > self.sum[idx << 1]
            value -= go_right * self.sum[idx << 1]
            idx = idx << 1 | go_right
        return idx - self.pow2
    
    def probs(self, idxs):
        return self.sum[idxs + self.pow2] / self.sum[1]


class SamplingPool(object):
    def __init__(self, args, S):
        self.S, self.L, self.total_items = S, 0, 0
        self.multi_step = args.multi_step
        
        self.prioritized_replay = "prioritized_replay" in args.enhancement
        if self.prioritized_replay:
            self.alpha = args.priority_alpha
            self.beta_0 = args.priority_beta
            self.max_P = 0
            self.sum_tree = SegmentTree(self.S)
    
    def add(self, idx, T):
        self.total_items += 1
        if self.L < self.S:
            self.L += 1
        if self.prioritized_replay:
            self.beta = self.beta_0 + (1 - self.beta_0) * (T / self.num_T)
            self.update_priority(self, np.array([idx]), np.array([self.max_P]))
    
    def update_priority(self, idx, P):
        self.max_P = max(self.max_P, P.max())
        P = np.power(P, self.alpha)
        self.sum_tree.update(idx, P)
    
    def sample(self, batch_size):
        for _ in range(20):
            if self.prior:
                values = np.random.rand(batch_size)
                idx = self.sum_tree.find(batch_size, values)
            else:
                idx = np.random.choice(self.L, batch_size, replace=True)
            if np.all(idx + self.multi_step < self.total_items):
                break
        if self.prioritized_replay:
            sampling_prob = self.sum_tree.probs(idx)
            importance_factor = (sampling_prob * self.L) ** -self.beta
        else:
            importance_factor = np.ones(batch_size)
        return importance_factor

        slice = np.arange(-self.window + 1, self.step_len + 1) + np.expand_dims(idxs, 1) # The k-th row of idx would be the adjacent observations of the k-th sampled transtion
        data = self.stack[slice % self.S] # To make it clear, always remember that data[:, self.window - 1] corresponds to current trinsition
        mask = np.zeros_like(data['done'], dtype=np.bool_) # Observartions in different episodes should not interact with each other; record those observations before or after current episode
        for i in range(self.window, self.window + self.step_len):
            mask[:, i] = np.logical_or(mask[:, i - 1], data['done'][:, i - 1])
        for i in range(self.window - 3, -1, -1):
            mask[:, i] = np.logical_or(mask[:, i + 1], data['done'][:, i]) # The observations before last episode end and strictly after current episode end are seen as empty observation
        data[mask] = self.empty_data
        if self.atari:
            states = torch.as_tensor(data['obs'][:, :self.window], dtype=torch.float32).to(device).div_(256)
            next_states = torch.as_tensor(data['obs'][:, self.step_len:self.step_len + self.window], dtype=torch.float32).to(device).div_(256)
        else:
            states = torch.tensor(data['obs'][:, :self.window]).to(device)
            next_states = torch.as_tensor(data['obs'][:, self.step_len:self.step_len + self.window]).to(device)
        actions = torch.as_tensor(data['act'][:, self.window - 1]).to(device)
        rewards = torch.as_tensor(data['reward'][:, self.window:self.window + self.step_len]).to(device)
        rewards = rewards @ self.gamma_vector
        dones = torch.as_tensor(mask[:, self.window + self.step_len - 1], dtype=torch.float32).to(device) # If next_state is belonged to another episode, current episode should be finished
        return states, actions, rewards, dones, next_states, idxs, weights # We need two more variables for priority buffer
