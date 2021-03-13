import torch.nn.functional as F
from torch import nn
import torch

class BatchNormCNN(nn.Module):
    def __init__(self):
        super(BatchNormCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=5)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv2_pool = nn.MaxPool2d(3)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=2, stride=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv3_pool = nn.MaxPool2d(3)

        self.flat = nn.Flatten()

        self.dense1 = nn.Linear(in_features=256, out_features=64)
        self.dense1_bn = nn.BatchNorm1d(64)

        self.dense2 = nn.Linear(64, 1)

    def forward(self, xb):
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2_pool(self.conv2_bn(self.conv2(xb))))
        xb = F.relu(self.conv3_pool(self.conv3_bn(self.conv3(xb))))
        xb = self.flat(xb)
        xb = F.relu(self.dense1_bn(self.dense1(xb)))
        xb = torch.sigmoid(self.dense2(xb))
        return xb
