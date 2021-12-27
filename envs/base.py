from abc import ABC


class BaseEnv(ABC):
    """Documentation
    The environment direct inherits OpenAI Gym API with these additional functions:
        1) explicit train/eval mode change
        2) explicit tell if the env is continuous
        3) maintain n_obs = observation shape,
                    n_act = action space dimension (or number of actions for discrete env)
           note!! one important thing here is to always use int-type n_obs if the observation
           is supposed to be a vector, because later we will assmue any env with n_obs in a
           tuple shape should be mathced with a CNN model rather than a MLP model
    
    We modify the action space so that:
        1) action \in [1, N] for a discrete environment
        2) action \in [-1, 1]^d for a continuous environment
    unless an additional declarification is made
    And we expect to get a one-dimensional numpy array as the input to the step method
    
    Method  |  Description
    ---------------------------------------------------------------------------
    reset   |  sample s_0 ~ μ_0; return o_0 = obs(s_0)
    step    |  s' ~ P(s'|s, a); return o' = obs(s'), r = R(s, a), done, information
    render  |  render a visible interface
    close   |  close the env and clean up
    train/eval           |  switching the env to train/eval mode
    continuous/discrete  |  tell whether the env is continuous/discrete
    """
    def __init__(self):
        super(BaseEnv, self).__init__()
        self.training = True
        self.continuous_action_space = False
        self.n_obs, self.n_act = None, None
    
    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError
        return next_obs, reward, done, env_info

    def render(self):
        raise NotImplementedError
        return image_visualization

    def close(self):
        raise NotImplementedError

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
    
    @property
    def continuous(self):
        return self.continuous_action_space

    @property
    def discrete(self):
        return not self.continuous_action_space
