from matplotlib import pyplot as plt

import torch

from tqdm import tqdm

from src.Data.Data import Data
from src.utils.utils import show_dataset
from src.utils.datautils import sample_from_data_loader


"""
Created collage of figurs from give directory
"""
DATA_PATH = "/Users/eoghanhogan/Desktop/Stage 4 Sem 2/Deep Learning/Project1.nosync/Project1/preprocessed_data_images"

def main():
    dev = torch.device("cpu")

    data = Data(DATA_PATH,
                augmented=False,
                workers=0,
                device=dev,
                batch_size=128)

    print("Creating Training Figure")
    xb, yb = sample_from_data_loader(data.get_train_data())
    f = show_dataset(xb, yb)
    f.suptitle('Subset of Eye Training Data', fontsize=30)
    f.savefig("./figs/eyes_train")

    print("Creating Validation Figure")
    xb, yb = sample_from_data_loader(data.get_val_data())
    f = show_dataset(xb, yb)
    f.suptitle('Subset of Eye Validation Data', fontsize=30)
    f.savefig("./figs/eyes_val")

    print("Creating Testing Figure")
    xb, yb = sample_from_data_loader(data.get_test_data())
    f = show_dataset(xb, yb)
    f.suptitle('Subset of Eye Testing Data', fontsize=30)
    f.savefig("./figs/eyes_test")

if __name__ == '__main__':
    main()
