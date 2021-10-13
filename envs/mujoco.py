import gym

from .base import BaseEnv


class MuJoCoRobot(BaseEnv):
    """OpenAI MuJoCo environment wrapper
    """
    
    def __init__(self, env, seed):
        super(MuJoCoRobot, self).__init__()
        self.env = gym.make(env)
        self.env.seed(seed)
        action_space = self.env.action_space
        self.continuous_action_space = True
        self.n_act = action_space.shape[0]
        self.action_scale = 0.5 * (action_space.high - action_space.low)
        self.action_mu = 0.5 * (action_space.high + action_space.low)
        obs_space = self.env.observation_space
        self.n_obs = obs_space.shape

    def reset(self):
        obs = self.env.reset()
        return obs
    
    def step(self, action):
        action = self.action_mu + action * self.action_scale
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()