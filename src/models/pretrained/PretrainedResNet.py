import torch
from torch import nn
import torchvision.models as models


def get_pretrained_ResNet(rn_type, classes, mode):
    model = None
    if rn_type == "18":
        model = models.resnet18(pretrained=True)
    elif rn_type == "34":
        model = models.resnet34(pretrained=True)
    elif rn_type == "50":
        model = models.resnet50(pretrained=True)
    elif rn_type == "101":
        model = models.resnet101(pretrained=True)
    elif rn_type == "152":
        model = models.resnet152(pretrained=True)
    elif rn_type == "50Wide":
        model = models.wide_resnet50_2(pretrained=True)
    elif rn_type == "101Wide":
        model = models.wide_resnet101_2(pretrained=True)
    if mode not in ["feature", "finetuning"]:
        raise RuntimeError("mode must be 'feature' or 'finetuning' ")
    if mode == "feature":
        for param in model.parameters():
            param.requires_grad = False
    model.fc = torch.nn.Sequential(
        nn.Linear(model.fc.in_features, classes, bias=True),
        nn.Sigmoid()
    )
    return model


def PretrainedResNet18(classes=1, mode="feature"):
    return get_pretrained_ResNet("18", classes, mode)


def PretrainedResNet34(classes=1, mode="feature"):
    return get_pretrained_ResNet("34", classes, mode)


def PretrainedResNet50(classes=1, mode="feature"):
    return get_pretrained_ResNet("50", classes, mode)


def PretrainedResNet101(classes=1, mode="feature"):
    return get_pretrained_ResNet("101", classes, mode)


def PretrainedResNet152(classes=1, mode="feature"):
    return get_pretrained_ResNet("152", classes, mode)


def PretrainedWideResNet50(classes=1, mode="feature"):
    return get_pretrained_ResNet("50Wide", classes, mode)


def PretrainedWideResNet101(classes=1, mode="feature"):
    return get_pretrained_ResNet("101Wide", classes, mode)
