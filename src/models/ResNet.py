from torch import nn
import torch

class Block(nn.Module):
    '''
    ResNet Block class
    '''

    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.expansion = 4

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)

        self.relu = nn.ReLU()

        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):

    def __init__(self, layers, in_channels, classes):
        # layers -  list to create the blocks
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_block_layer_(layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_block_layer_(layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_block_layer_(layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_block_layer_(layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.flat = nn.Flatten()

        self.dense1 = nn.Linear(512*4, classes)

    def _make_block_layer_(self, residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []
        if (stride != 1) or (self.in_channels != (out_channels * 4)):
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels*4))
        layers.append(Block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels*4
        for i in range(residual_blocks - 1):
            layers.append(Block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.flat(x)
        x = self.dense1(x)
        x = torch.sigmoid(x)
        return x


def ResNet50(in_channels=3, classes=2):
    return ResNet([3, 4, 6, 3], in_channels, classes)


def ResNet101(in_channels=3, classes=2):
    return ResNet([3, 4, 23, 3], in_channels, classes)


def ResNet152(in_channels=3, classes=2):
    return ResNet([3, 8, 36, 3], in_channels, classes)
