from matplotlib import pyplot as plt

import torch

from tqdm import tqdm

from src.Data.Data import Data
from src.utils.utils import show_dataset

DATA_PATH = "/Users/eoghanhogan/Desktop/Stage 4 Sem 2/Deep Learning/Project1.nosync/Project1/preprocessed_data_images"

def main():
    dev = torch.device("cpu")

    data = Data(DATA_PATH,
                workers=0,
                dev=dev,
                batch_size=128)

    print("Creating Training Figure")
    xb, yb = [(x, y) for x, y in tqdm(data.get_train_data(), "loading training data")][0]
    f = show_dataset(xb, yb)
    f.suptitle('Subset of Eye Training Data', fontsize=30)
    f.savefig("./figs/eyes_train")

    print("Creating Validation Figure")
    xb, yb = [(x, y) for x, y in tqdm(data.get_val_data(), "loading validation data")][0]
    f = show_dataset(xb, yb)
    f.suptitle('Subset of Eye Validation Data', fontsize=30)
    f.savefig("./figs/eyes_val")

    print("Creating Testing Figure")
    xb, yb = [(x, y) for x, y in tqdm(data.get_test_data(), "loading testing data")][0]
    f = show_dataset(xb, yb)
    f.suptitle('Subset of Eye Testing Data', fontsize=30)
    f.savefig("./figs/eyes_test")

if __name__ == '__main__':
    main()
