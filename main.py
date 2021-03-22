from preprocess import main as pm
from augment import main as am

from src.experiments.experiments import BasicCNNExpt
from src.experiments.experiments import BatchNormCNNExpt
from src.experiments.experiments import VGG11Expt
from src.experiments.experiments import ResNet50Expt
from src.experiments.experiments import ResNet152Expt
from src.experiments.experiments import EfficientNetB0Expt
from src.experiments.experiments import EfficientNetB3Expt


"""
Run everything together
WARNING THIS COULD TAKE DAYS and fail if not enough GPU/system memory
RECCOMMENDED USE IS TO TAKE ONE EXPERIMENT FROM THE EXPERIMENTS PACKAGE AND RUN IT.
THIS FILE RUNS EVERYTHING I DID THROUGHOUT THE PROJECT ALL AT ONCE
"""

non_aug_directories = {
    "data": "./preprocessed_data_images",
    "ckp": "./checkpoints",
    "log": "./logs"
}

aug_directories = {
    "data": "./augmented",
    "ckp": "./checkpoints",
    "log": "./logs"
}

WORKERS = 0


def main():
    # preprocess Data
    pm()
    # Augment Data
    am()

    # RUN EXPERIMENTS
    BasicCNNExpt(non_aug_directories, WORKERS, False)
    BatchNormCNNExpt(non_aug_directories, WORKERS, False)
    BasicCNNExpt(aug_directories, WORKERS, True)
    BatchNormCNNExpt(aug_directories, WORKERS, True)
    VGG11Expt(aug_directories, WORKERS, True)
    ResNet50Expt(aug_directories, WORKERS, True)
    ResNet152Expt(aug_directories, WORKERS, True)
    EfficientNetB0Expt(aug_directories, WORKERS, True)
    EfficientNetB3Expt(aug_directories, WORKERS, True)

    # Final Model
    EfficientNetB0Expt(aug_directories,
                       WORKERS,
                       True,
                       save_every=10,
                       lrs=[0.0001],
                       bss=[64],
                       total_amt=65_536,
                       val_percent=0.1,
                       test_amt=1000,
                       train_early_stopping=True,
                       test_early_stopping=False,
                       early_stopping_attention=5)


if __name__ == '__main__':
    main()
