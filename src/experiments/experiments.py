from src.experiments.RunExperiment import RunExpt
from src.experiments.utils import get_resize_wrapper

from src.models.BasicCNN import BasicCNN
from src.models.BatchNormCNN import BatchNormCNN
from src.models.VGG_net import VGG11, VGG16
from src.models.ResNet import ResNet50, ResNet101, ResNet152
from src.models.EfficientNet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7

import torch


def base_experiment(name, model, directories, size, workers, kwargs, aug):
    lrs = [0.1, 0.003, 0.0008, 0.0001]
    bss = [32, 64]
    opts = [torch.optim.Adam]
    losses = [torch.nn.BCELoss]
    wrapper = get_resize_wrapper(size)
    RunExpt(f"{name}_EXPT", model, kwargs, 100, directories, aug, wrapper, lrs=lrs, bss=bss, opts=opts, losses=losses, workers=workers)

def BasicCNNExpt(directories, workers, aug):
    kwargs = {}
    base_experiment(f"BasicCNN{'AUG' if aug else ''}", BasicCNN, directories, 512, workers, kwargs, aug)

def BatchNormCNNExpt(directories, workers, aug):
    kwargs = {}
    base_experiment(f"BatchNorm{'AUG' if aug else ''}", BatchNormCNN, directories, 256, workers, kwargs, aug)

def VGG11Expt(directories, workers, aug):
    kwargs = {"in_channels":3, "output_size":1}
    base_experiment(f"VGG11{'AUG' if aug else ''}", VGG11, directories, 244, workers, kwargs, aug)

def ResNet50Expt(directories, workers, aug):
    kwargs = {"in_channels": 3, "classes": 1}
    base_experiment(f"ResNet50{'AUG' if aug else ''}", ResNet50, directories, 244, workers, kwargs, aug)

def ResNet101Expt(directories, workers, aug):
    kwargs = {"in_channels": 3, "classes": 1}
    base_experiment(f"ResNet101{'AUG' if aug else ''}", ResNet101, directories, 244, workers, kwargs, aug)

def EfficientNetB0Expt(directories, workers, aug):
    kwargs = {"classes": 1}
    base_experiment(f"EfficientNetB0{'AUG' if aug else ''}", EfficientNetB0, directories, 244, workers, kwargs, aug)

def EfficientNetB1Expt(directories, workers, aug):
    kwargs = {"classes": 1}
    base_experiment(f"EfficientNetB1{'AUG' if aug else ''}", EfficientNetB1, directories, 240, workers, kwargs, aug)

def EfficientNetB2Expt(directories, workers, aug):
    kwargs = {"classes": 1}
    base_experiment(f"EfficientNetB2{'AUG' if aug else ''}", EfficientNetB2, directories, 260, workers, kwargs, aug)

def EfficientNetB3Expt(directories, workers, aug):
    kwargs = {"classes": 1}
    base_experiment(f"EfficientNetB3{'AUG' if aug else ''}", EfficientNetB3, directories, 300, workers, kwargs, aug)

