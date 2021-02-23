import torch
import torch.nn.functional as F
from torch import nn

import numpy as np

class Basic_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=1)

        self.flat = nn.Flatten()

        self.feed1 = nn.Linear(4096, 64)
        self.feed2 = nn.Linear(64, 1)

    def convAndPool(self, xb):
        xb = F.relu(self.conv1(xb))
        xb = F.max_pool2d(xb, (2, 2))
        xb = F.relu(self.conv2(xb))
        xb = F.max_pool2d(xb, (2, 2))
        xb = F.relu(self.conv3(xb))
        xb = F.max_pool2d(xb, (2, 2))
        xb = self.flat(xb)
        return xb

    def feedForward(self, xb):
        xb = F.relu(self.feed1(xb))
        xb = torch.sigmoid(self.feed2(xb))
        return xb

    def forward(self, xb):
        xb = self.convAndPool(xb)
        xb = self.feedForward(xb)
        return xb
