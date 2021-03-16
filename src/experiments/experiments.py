from src.experiments.RunExperiment import RunExpt
from src.experiments.utils import get_resize_wrapper
from src.models.BasicCNN import BasicCNN
from src.models.BatchNormCNN import BatchNormCNN
from src.models.VGG_net import VGG11, VGG16
from src.models.ResNet import ResNet50, ResNet101, ResNet152
from src.models.EfficientNet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7

import torch


def base_experiment(name, model, directories, size, workers, model_kwargs, aug, save_every=5):
    """
    Run a single experiment
    :param name: name of experiment
    :param model: the class of the model to use
    :param directories: a dictionary of the data, logs, and checkpoints
    :param size: the size to scale the images to so that the network will accept them
    :param workers: the number of workers to use for data loading
    :param model_kwargs: the arguments to pass to the model when initialized
    :param aug: wether ot not the data is augmented
    :param save_every: how often to save the model
    """
    lrs = [0.1, 0.003, 0.0008, 0.0001]
    bss = [32, 64]
    opts = [torch.optim.Adam]
    losses = [torch.nn.BCELoss]
    wrapper = get_resize_wrapper(size)
    RunExpt(f"{name}_EXPT", model, model_kwargs, 100, directories, aug, wrapper, lrs=lrs, bss=bss, opts=opts, losses=losses, workers=workers, save_every=save_every)


def BasicCNNExpt(directories, workers, aug, save_every=5):
    model_kwargs = {}
    base_experiment(f"BasicCNN{'AUG' if aug else ''}", BasicCNN, directories, 512, workers, model_kwargs, aug, save_every=save_every)


def BatchNormCNNExpt(directories, workers, aug, save_every=5):
    model_kwargs = {}
    base_experiment(f"BatchNorm{'AUG' if aug else ''}", BatchNormCNN, directories, 256, workers, model_kwargs, aug, save_every=save_every)


def VGG11Expt(directories, workers, aug, save_every=5):
    model_kwargs = {"in_channels":3, "output_size":1}
    base_experiment(f"VGG11{'AUG' if aug else ''}", VGG11, directories, 244, workers, model_kwargs, aug, save_every=save_every)


def VGG16Expt(directories, workers, aug, save_every=5):
    model_kwargs = {"in_channels":3, "output_size":1}
    base_experiment(f"VGG16{'AUG' if aug else ''}", VGG16, directories, 244, workers, model_kwargs, aug, save_every=save_every)


def ResNet50Expt(directories, workers, aug, save_every=5):
    model_kwargs = {"in_channels": 3, "classes": 1}
    base_experiment(f"ResNet50{'AUG' if aug else ''}", ResNet50, directories, 244, workers, model_kwargs, aug, save_every=save_every)


def ResNet101Expt(directories, workers, aug, save_every=5):
    model_kwargs = {"in_channels": 3, "classes": 1}
    base_experiment(f"ResNet101{'AUG' if aug else ''}", ResNet101, directories, 244, workers, model_kwargs, aug, save_every=save_every)


def ResNet152Expt(directories, workers, aug, save_every=5):
    model_kwargs = {"in_channels": 3, "classes": 1}
    base_experiment(f"ResNet152{'AUG' if aug else ''}", ResNet152, directories, 244, workers, model_kwargs, aug, save_every=save_every)


def EfficientNetB0Expt(directories, workers, aug, save_every=5):
    model_kwargs = {"classes": 1}
    base_experiment(f"EfficientNetB0{'AUG' if aug else ''}", EfficientNetB0, directories, 244, workers, model_kwargs, aug, save_every=save_every)


def EfficientNetB1Expt(directories, workers, aug, save_every=5):
    model_kwargs = {"classes": 1}
    base_experiment(f"EfficientNetB1{'AUG' if aug else ''}", EfficientNetB1, directories, 240, workers, model_kwargs, aug, save_every=save_every)


def EfficientNetB2Expt(directories, workers, aug, save_every=5):
    model_kwargs = {"classes": 1}
    base_experiment(f"EfficientNetB2{'AUG' if aug else ''}", EfficientNetB2, directories, 260, workers, model_kwargs, aug, save_every=save_every)


def EfficientNetB3Expt(directories, workers, aug, save_every=5):
    model_kwargs = {"classes": 1}
    base_experiment(f"EfficientNetB3{'AUG' if aug else ''}", EfficientNetB3, directories, 300, workers, model_kwargs, aug, save_every=save_every)


def EfficientNetB4Expt(directories, workers, aug, save_every=5):
    model_kwargs = {"classes": 1}
    base_experiment(f"EfficientNetB4{'AUG' if aug else ''}", EfficientNetB4, directories, 300, workers, model_kwargs, aug, save_every=save_every)


def EfficientNetB5Expt(directories, workers, aug, save_every=5):
    model_kwargs = {"classes": 1}
    base_experiment(f"EfficientNetB5{'AUG' if aug else ''}", EfficientNetB5, directories, 300, workers, model_kwargs, aug, save_every=save_every)


def EfficientNetB6Expt(directories, workers, aug, save_every=5):
    model_kwargs = {"classes": 1}
    base_experiment(f"EfficientNetB6{'AUG' if aug else ''}", EfficientNetB6, directories, 300, workers, model_kwargs, aug, save_every=save_every)


def EfficientNetB7Expt(directories, workers, aug, save_every=5):
    model_kwargs = {"classes": 1}
    base_experiment(f"EfficientNetB7{'AUG' if aug else ''}", EfficientNetB7, directories, 300, workers, model_kwargs, aug, save_every=save_every)

