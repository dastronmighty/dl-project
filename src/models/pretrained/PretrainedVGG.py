from torch import nn
import torchvision.models as models


def get_pretrained_VGG(rn_type, classes, mode):
    if mode not in ["feature", "finetuning"]:
        raise RuntimeError("mode must be 'feature' or 'finetuning' ")
    model = None
    if rn_type == "11":
        model = models.vgg11(pretrained=True)
    elif rn_type == "13":
        model = models.vgg13(pretrained=True)
    elif rn_type == "16":
        model = models.vgg16_bn(pretrained=True)
    elif rn_type == "19":
        model = models.vgg16_bn(pretrained=True)
    elif rn_type == "11bn":
        model = models.vgg11_bn(pretrained=True)
    elif rn_type == "13bn":
        model = models.vgg13_bn(pretrained=True)
    elif rn_type == "16bn":
        model = models.vgg16_bn(pretrained=True)
    elif rn_type == "19bn":
        model = models.vgg19_bn(pretrained=True)
    if mode == "feature":
        for param in model.parameters():
            param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096, bias=True),
        nn.ReLU(),
        nn.Linear(4096, 1000, bias=True),
        nn.ReLU(),
        nn.Linear(1000, classes, bias=True),
        nn.Sigmoid()
    )
    return model


def PretrainedVGG11(classes=1, mode="feature"):
    return get_pretrained_VGG("11", classes, mode)


def PretrainedVGG11BN(classes=1, mode="feature"):
    return get_pretrained_VGG("11bn", classes, mode)


def PretrainedVGG13(classes=1, mode="feature"):
    return get_pretrained_VGG("13", classes, mode)


def PretrainedVGG13BN(classes=1, mode="feature"):
    return get_pretrained_VGG("13bn", classes, mode)


def PretrainedVGG16(classes=1, mode="feature"):
    return get_pretrained_VGG("16", classes, mode)


def PretrainedVGG16BN(classes=1, mode="feature"):
    return get_pretrained_VGG("16bn", classes, mode)


def PretrainedVGG19(classes=1, mode="feature"):
    return get_pretrained_VGG("19", classes, mode)


def PretrainedVGG19BN(classes=1, mode="feature"):
    return get_pretrained_VGG("19bn", classes, mode)
