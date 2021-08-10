import torch.nn as nn
import torch.nn.functional as F


class CNNFeature(nn.Module):
    def __init__(self):
        pass

class MLPFeature(nn.Module):
    def __init__(self):
        pass

class Critic(nn.Module):
    def __init__(self, args, n_obs):
        super(Critic, self).__init__()
        self.value = nn.Linear(args.hidden_nodes, 1)

class PPOAgent(object):
    def __init__(self, args):
        print("PPO Agent position: {}".format(__name__))
        print("bs = %d" % args.batch_size)