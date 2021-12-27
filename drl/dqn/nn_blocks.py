import torch.nn as nn


class MLPFeature(nn.Sequential):
    def __init__(self, input_dim, hidden_dim):
        super(MLPFeature, self).__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

class CNNFeature(nn.Sequential):
    def __init__(self, args, in_channels, width, height, channel_dim):
        divider_list = args.channel_divider
        kernel_list = args.kernel_size
        stride_list = args.stride
        layers = []
        for divider, kernel, stride in zip(divider_list, kernel_list, stride_list):
            layers.append(nn.Conv2d(in_channels, channel_dim // divider, kernel, stride))
            layers.append(nn.ReLU())
            in_channels = channel_dim // divider
            width = (width - kernel) // stride + 1
            height = (height - kernel) // stride + 1
        layers.append(nn.Flatten())
        self.conv_nodes = in_channels * width * height
        super(CNNFeature, self).__init__(*layers)

class QNet(nn.Module):
    def __init__(self, args, n_obs, n_act):
        super(QNet, self).__init__()
        self.hidden_dim = args.hidden_dim
        if len(n_obs) == 1:
            self.feature_extractor = MLPFeature(*n_obs, self.hidden_dim)
            self.feature_dim = self.hidden_dim
        else:
            self.feature_extractor = CNNFeature(args, *n_obs, args.channel_dim)
            self.feature_dim = self.feature_extractor.conv_nodes
        self.Q_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, n_act),
        )
    
    def forward(self, obs):
        feature = self.feature_extractor(obs)
        return self.Q_mlp(feature)
