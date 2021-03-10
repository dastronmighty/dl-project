import torch
import torch.nn.functional as F
from torch import nn


class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(5, 5), stride=3, padding=1)

        self.avgpool1 = nn.AvgPool2d(4, stride=2)

        self.flat = nn.Flatten()

        self.feed1 = nn.Linear(256, 16)
        self.feed2 = nn.Linear(16, 1)

    def forward(self, xb):
        xb = F.relu(self.conv1(xb))
        xb = self.avgpool1(xb)
        xb = F.relu(self.conv2(xb))
        xb = self.avgpool1(xb)
        xb = F.relu(self.conv3(xb))
        xb = self.avgpool1(xb)
        xb = self.flat(xb)
        xb = torch.sigmoid(self.feed1(xb))
        xb = F.softmax(self.feed2(xb), dim=0)
        return xb