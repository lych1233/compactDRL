import gym
import numpy as np
import torch

from envs import BaseEnv


class Control(BaseEnv):
    """OpenAI Gym Classic Control environment wrapper
    """  
    def __init__(self, env):
        self.env = gym.make(env)
        self.env.seed(np.random.randint(int(1e9)))
        action_space = self.env.action_space
        if isinstance(action_space, gym.spaces.Discrete):
            self.continuous_action_space = False
            self.n_act = action_space.n
        else:
            self.continuous_action_space = True
            self.n_act = action_space.shape[0]
            self.action_scale = 0.5 * (action_space.high - action_space.low)
            self.action_mu = 0.5 * (action_space.high + action_space.low)
        obs_space = self.env.observation_space
        self.n_obs = obs_space.shape[0]

    def reset(self):
        obs = self.env.reset()
        return obs
    
    def step(self, action):
        action = self.continuous_action_space(action, self.continuous)
        if self.continuous:
            action = self.action_mu + action * self.action_scale
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info
    
    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()
        
    def get_proper_action(self, action, is_continuous=False):
        if isinstance(action, (np.ndarray, torch.Tensor)):
            action = action.item()
        if isinstance(action, (list, tuple)):
            action = np.array(action).item()
        if not is_continuous and isinstance(action, int):
            return action
        elif is_continuous and isinstance(action, float):
            return action
        else:
            raise TypeError("Only single value with type [int/float, np.ndarray, torch.Tensor, list, tuple] is allowed")