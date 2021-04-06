import torch
from torch import nn
import torchvision.models as models


def get_pretrained_ResNet(rn_type):
    model = None
    if rn_type == "50":
        model = models.resnet50(pretrained=True)
    elif rn_type == "101":
        model = models.resnet101(pretrained=True)
    elif rn_type == "152":
        model = models.resnet152(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = torch.nn.Sequential(
        nn.Linear(2048, 1, bias=True),
        nn.Sigmoid()
    )
    return model


def PretrainedResNet50():
    return get_pretrained_ResNet("50")


def PretrainedResNet101():
    return get_pretrained_ResNet("101")


def PretrainedResNet152():
    return get_pretrained_ResNet("152")