from src.experiments.RunExperiment import pretrained_experiment

from src.models.pretrained.PretrainedInception import PretrainedInceptionV3
from src.models.pretrained.PretrainedVGG import *
from src.models.pretrained.PretrainedResNet import *


def PretrainedExperimentVGG11(directories, workers, aug,  extra_tag="", model_kwargs={"mode" : "feature"}, **kwargs):
    pretrained_experiment(f"{extra_tag}PretrainedExperimentVGG11{'AUG' if aug else ''}", PretrainedVGG11, directories, 244, workers, model_kwargs, aug,  **kwargs)


def PretrainedExperimentVGG11BN(directories, workers, aug,  extra_tag="", model_kwargs={"mode" : "feature"}, **kwargs):
     
    pretrained_experiment(f"{extra_tag}PretrainedExperimentVGG11BN{'AUG' if aug else ''}", PretrainedVGG11BN, directories, 244, workers, model_kwargs, aug,  **kwargs)


def PretrainedExperimentVGG13(directories, workers, aug,  extra_tag="", model_kwargs={"mode":"feature"}, **kwargs):
     
    pretrained_experiment(f"{extra_tag}PretrainedExperimentVGG13{'AUG' if aug else ''}", PretrainedExperimentVGG13, directories, 244,
                          workers, model_kwargs, aug,  **kwargs)


def PretrainedExperimentVGG13BN(directories, workers, aug,  extra_tag="", model_kwargs={"mode":"feature"}, **kwargs):
     
    pretrained_experiment(f"{extra_tag}PretrainedExperimentVGG13BN{'AUG' if aug else ''}", PretrainedExperimentVGG13BN, directories, 244, workers, model_kwargs, aug,  **kwargs)
        

def PretrainedExperimentVGG16(directories, workers, aug,  extra_tag="", model_kwargs={"mode":"feature"}, **kwargs):
     
    pretrained_experiment(f"{extra_tag}PretrainedExperimentVGG16{'AUG' if aug else ''}", PretrainedExperimentVGG16, directories, 244, workers, model_kwargs, aug,  **kwargs)


def PretrainedExperimentVGG16BN(directories, workers, aug,  extra_tag="", model_kwargs={"mode":"feature"}, **kwargs):
     
    pretrained_experiment(f"{extra_tag}PretrainedExperimentVGG16BN{'AUG' if aug else ''}", PretrainedExperimentVGG16BN, directories, 244, workers, model_kwargs, aug,  **kwargs)


def PretrainedExperimentVGG19(directories, workers, aug,  extra_tag="", model_kwargs={"mode":"feature"}, **kwargs):
     
    pretrained_experiment(f"{extra_tag}PretrainedExperimentVGG19{'AUG' if aug else ''}", PretrainedExperimentVGG19, directories, 244, workers, model_kwargs, aug,  **kwargs)


def PretrainedExperimentVGG19BN(directories, workers, aug,  extra_tag="", model_kwargs={"mode":"feature"}, **kwargs):
     
    pretrained_experiment(f"{extra_tag}PretrainedExperimentVGG19BN{'AUG' if aug else ''}", PretrainedExperimentVGG19BN, directories, 244, workers, model_kwargs, aug,  **kwargs)


def PretrainedExperimentInceptionV3(directories, workers, aug,  extra_tag="", model_kwargs={"mode":"feature"}, **kwargs):
     
    pretrained_experiment(f"{extra_tag}PretrainedExperimentInceptionV3{'AUG' if aug else ''}", PretrainedInceptionV3, directories, 299, workers, model_kwargs, aug,  **kwargs)


def PretrainedExperimentResNet18(directories, workers, aug,  extra_tag="", model_kwargs={"mode":"feature"}, **kwargs):
     
    pretrained_experiment(f"{extra_tag}PretrainedExperimentResNet18{'AUG' if aug else ''}", PretrainedExperimentResNet18, directories, 244, workers, model_kwargs, aug,  **kwargs)


def PretrainedExperimentResNet34(directories, workers, aug,  extra_tag="", model_kwargs={"mode":"feature"}, **kwargs):
     
    pretrained_experiment(f"{extra_tag}PretrainedExperimentResNet34{'AUG' if aug else ''}", PretrainedExperimentResNet34, directories, 244, workers, model_kwargs, aug,  **kwargs)


def PretrainedExperimentResNet50(directories, workers, aug,  extra_tag="", model_kwargs={"mode":"feature"}, **kwargs):
     
    pretrained_experiment(f"{extra_tag}PretrainedExperimentResNet50{'AUG' if aug else ''}", PretrainedExperimentResNet50, directories, 244, workers, model_kwargs, aug,  **kwargs)


def PretrainedExperimentResNet101(directories, workers, aug,  extra_tag="", model_kwargs={"mode":"feature"}, **kwargs):
     
    pretrained_experiment(f"{extra_tag}PretrainedExperimentResNet101{'AUG' if aug else ''}", PretrainedExperimentResNet101, directories, 244, workers, model_kwargs, aug,  **kwargs)


def PretrainedExperimentResNet152(directories, workers, aug,  extra_tag="", model_kwargs={"mode":"feature"}, **kwargs):
     
    pretrained_experiment(f"{extra_tag}PretrainedExperimentResNet152{'AUG' if aug else ''}", PretrainedExperimentResNet152, directories, 244, workers, model_kwargs, aug,  **kwargs)


def PretrainedExperimentWideResNet50(directories, workers, aug,  extra_tag="", model_kwargs={"mode":"feature"}, **kwargs):
     
    pretrained_experiment(f"{extra_tag}PretrainedExperimentWideResNet50{'AUG' if aug else ''}", PretrainedWideResNet50, directories, 244, workers, model_kwargs, aug,  **kwargs)


def PretrainedExperimentWideResNet101(directories, workers, aug,  extra_tag="", model_kwargs={"mode":"feature"}, **kwargs):
     
    pretrained_experiment(f"{extra_tag}PretrainedExperimentWideResNet101{'AUG' if aug else ''}", PretrainedWideResNet101, directories, 244, workers, model_kwargs, aug,  **kwargs)
