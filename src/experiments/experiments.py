from src.experiments.RunExperiment import regular_experiment

from src.models.BasicCNN import BasicCNN
from src.models.BatchNormCNN import BatchNormCNN
from src.models.VGG_net import VGG11, VGG16
from src.models.ResNet import ResNet50, ResNet101, ResNet152
from src.models.EfficientNet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7


def BasicCNNExpt(directories, workers, aug, **kwargs):
    model_kwargs = {}
    regular_experiment(f"BasicCNN{'AUG' if aug else ''}", BasicCNN, directories, 512, workers, model_kwargs, aug, **kwargs)


def BatchNormCNNExpt(directories, workers, aug, **kwargs):
    model_kwargs = {}
    regular_experiment(f"BatchNorm{'AUG' if aug else ''}", BatchNormCNN, directories, 256, workers, model_kwargs, aug, **kwargs)


def VGG11Expt(directories, workers, aug, **kwargs):
    model_kwargs = {"in_channels" : 3, "output_size" : 1}
    regular_experiment(f"VGG11{'AUG' if aug else ''}", VGG11, directories, 244, workers, model_kwargs, aug, **kwargs)


def VGG16Expt(directories, workers, aug, **kwargs):
    model_kwargs = {"in_channels":3, "output_size":1}
    regular_experiment(f"VGG16{'AUG' if aug else ''}", VGG16, directories, 244, workers, model_kwargs, aug, **kwargs)


def ResNet50Expt(directories, workers, aug, **kwargs):
    model_kwargs = {"in_channels": 3, "classes": 1}
    regular_experiment(f"ResNet50{'AUG' if aug else ''}", ResNet50, directories, 244, workers, model_kwargs, aug, **kwargs)


def ResNet101Expt(directories, workers, aug, **kwargs):
    model_kwargs = {"in_channels": 3, "classes": 1}
    regular_experiment(f"ResNet101{'AUG' if aug else ''}", ResNet101, directories, 244, workers, model_kwargs, aug, **kwargs)


def ResNet152Expt(directories, workers, aug, **kwargs):
    model_kwargs = {"in_channels": 3, "classes": 1}
    regular_experiment(f"ResNet152{'AUG' if aug else ''}", ResNet152, directories, 244, workers, model_kwargs, aug, **kwargs)


def EfficientNetB0Expt(directories, workers, aug, **kwargs):
    model_kwargs = {"classes": 1}
    regular_experiment(f"EfficientNetB0{'AUG' if aug else ''}", EfficientNetB0, directories, 244, workers, model_kwargs, aug, **kwargs)


def EfficientNetB1Expt(directories, workers, aug, **kwargs):
    model_kwargs = {"classes": 1}
    regular_experiment(f"EfficientNetB1{'AUG' if aug else ''}", EfficientNetB1, directories, 240, workers, model_kwargs, aug, **kwargs)


def EfficientNetB2Expt(directories, workers, aug, **kwargs):
    model_kwargs = {"classes": 1}
    regular_experiment(f"EfficientNetB2{'AUG' if aug else ''}", EfficientNetB2, directories, 260, workers, model_kwargs, aug, **kwargs)


def EfficientNetB3Expt(directories, workers, aug, **kwargs):
    model_kwargs = {"classes": 1}
    regular_experiment(f"EfficientNetB3{'AUG' if aug else ''}", EfficientNetB3, directories, 300, workers, model_kwargs, aug, **kwargs)


def EfficientNetB4Expt(directories, workers, aug, **kwargs):
    model_kwargs = {"classes": 1}
    regular_experiment(f"EfficientNetB4{'AUG' if aug else ''}", EfficientNetB4, directories, 380, workers, model_kwargs, aug, **kwargs)


def EfficientNetB5Expt(directories, workers, aug, **kwargs):
    model_kwargs = {"classes": 1}
    regular_experiment(f"EfficientNetB5{'AUG' if aug else ''}", EfficientNetB5, directories, 457, workers, model_kwargs, aug, **kwargs)


def EfficientNetB6Expt(directories, workers, aug, **kwargs):
    model_kwargs = {"classes": 1}
    regular_experiment(f"EfficientNetB6{'AUG' if aug else ''}", EfficientNetB6, directories, 528, workers, model_kwargs, aug, **kwargs)


def EfficientNetB7Expt(directories, workers, aug, **kwargs):
    model_kwargs = {"classes": 1}
    regular_experiment(f"EfficientNetB7{'AUG' if aug else ''}", EfficientNetB7, directories, 600, workers, model_kwargs, aug, **kwargs)

