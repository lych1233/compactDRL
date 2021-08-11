class BaseEnv(object):
    """Documentation
    The environment direct inherits OpenAI Gym API with these additional functions:
        1) explicit train/eval mode change
        2) explicit tell if the env is continuous
        3) maintain n_obs = observation shape,
                    n_act = action space dimension (or number of actions in discrete case)
           note!! one important thing here is to always use int-type n_obs if the observation
           is supposed to be a vector, because later we will assmue any env with n_obs in a
           tuple shape should be mathced with a CNN model rather than a MLP model
    
    We modify the action space so that:
        1) action \in [1, N] for a discrete environment
        2) action \in [-1, 1]^d for a continuous environment
    unless an additional declarification is made

    You may transfer any type of actions (int/float, np.ndarray, torch.Tensor, list, tuple)
    and the environment will change it to a proper one if possible
    
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
        self.training = True
        self.continuous_action_space = False
        self.n_obs, self.n_act = None, None
    
    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
    
    @property
    def continuous(self):
        return self.continuous_action_space

    @property
    def discrete(self):
        return not self.continuous_action_space

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
    
    @staticmethod
    def get_proper_action(action):
        return action
