import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.categorical import Categorical

from env import Env

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )

class ResidualBlock(nn.Module):
    def __init__(self, num_channels, stride=1):
        super().__init__()
        self.conv1 = conv3x3(num_channels, num_channels, stride)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = conv3x3(num_channels, num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = F.relu(out)
        return out


class Network(nn.Module):
    def __init__(self, board_area):
        super().__init__()

        self.board_area = board_area

        self.conv = conv3x3(4, 128)
        self.bn = nn.BatchNorm2d(128)

        self.res_blocks = [ResidualBlock(128) for i in range(6)]

        self.policy_conv1x1 = nn.Conv2d(128, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2*self.board_area, self.board_area)

        self.value_conv1x1 = nn.Conv2d(128, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1*self.board_area, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)

        for block in self.res_blocks:
            x = block(x)

        pi = self.policy_conv1x1(x)
        pi = self.policy_bn(pi)
        pi = F.relu(pi)
        pi = pi.view(-1, 2*self.board_area)
        pi = self.policy_fc(pi)
        pi = F.log_softmax(pi, dim=1)

        v = self.value_conv1x1(x)
        v = self.value_bn(v)
        v = F.relu(v)
        v = v.view(-1, 1*self.board_area)
        v = self.value_fc1(v)
        v = F.relu(v)
        v = self.value_fc2(v)
        v = torch.tanh(v)

        return pi, v

class Model():
    def __init__(self, board_area, device="cpu"):
        self.board_area = board_area
        self.device = device
        
        self.net = Network(self.board_area).to(self.device)

    def step(self, env: Env):
        with torch.no_grad():
            obs = torch.from_numpy(env.get_observation()).unsqueeze(0).to(self.device)
            log_pi, value = self.net(obs)
            log_pi = log_pi.cpu().numpy().flatten()
            pi = np.exp(log_pi)
            pi = zip(env.available, pi[env.available])
            value = value.item()
        
        return pi, value

    def batch_step(self, batch_obs):
        log_pi, value = self.net(batch_obs.to(self.device))

        return log_pi.cpu(), value.view(-1).cpu()

    def load(self, weights):
        self.net.load_state_dict(weights)

    def weights(self):
        return self.net.state_dict()

    def parameters(self):
        return self.net.parameters()