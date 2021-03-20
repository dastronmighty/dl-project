from matplotlib import pyplot as plt

import torch

from tqdm import tqdm

from src.Data.Data import Data
from src.utils.utils import show_dataset
from src.utils.datautils import sample_from_data_loader

"""
Created collage of figures from give directory
"""

DATA_PATH = "/Users/eoghanhogan/Desktop/Stage 4 Sem 2/Deep Learning/Project1.nosync/Project1/preprocessed_data_images"
DATA_AUG_PATH = "/Users/eoghanhogan/Desktop/Stage 4 Sem 2/Deep Learning/Project1.nosync/Project1/augmented"
FIG_DIR = "./figs"


def data_to_figures(data, save_to, name):
    print("Creating Training Figure")
    xb, yb = sample_from_data_loader(data.get_train_data())
    f = show_dataset(xb, yb)
    f.suptitle('Subset of Eye Training Data', fontsize=30)
    f.savefig(f"{save_to}/{name}_eyes_train")

    print("Creating Validation Figure")
    xb, yb = sample_from_data_loader(data.get_val_data())
    f = show_dataset(xb, yb)
    f.suptitle('Subset of Eye Validation Data', fontsize=30)
    f.savefig(f"{save_to}/{name}_eyes_val")

    print("Creating Testing Figure")
    xb, yb = sample_from_data_loader(data.get_test_data())
    f = show_dataset(xb, yb)
    f.suptitle('Subset of Eye Testing Data', fontsize=30)
    f.savefig(f"{save_to}/{name}_eyes_test")


def main():
    dev = torch.device("cpu")

    data = Data(DATA_PATH,
        augmented=False,
        workers=0,
        device=dev,
        batch_size=128,
        verbose=True)

    data_to_figures(data, FIG_DIR, "regular")

    data = Data(DATA_AUG_PATH,
                augmented=True,
                workers=0,
                device=dev,
                batch_size=128,
                verbose=True)

    data_to_figures(data, FIG_DIR, "augmented")


if __name__ == '__main__':
    main()
