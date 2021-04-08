import torch
import torchvision
import numpy as np
import random
import os

from torch.utils.data import Dataset

from src.Data.Data import Data


def seed_worker(worker_id):
    """
    Seed a worker (if using). This is provided by pytorch as how to ensure reporducibility
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CustomDataset(Dataset):
    """
    This is a custom dataset we will use for this project it deals specifically with folders of images as data.
    loading all the images as pytorch tensors is not possible due to the size required
    also needing to be able to change sizes on the fly means we cant pre split tensors and use them
    the solution is a custom dataset for loading images to tensors
    """

    def __init__(self, files):
        """
        :param files: the list of paths
        """
        self.files = files

    def preprocess(self, file):
        """
        turn file path to lable and tensor
        :param file: the file
        :return: image tensor as x, label as y
        """
        y = int(file[-5:-4])
        img = torchvision.io.read_image(file)
        x = img.float()
        return x, y

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.preprocess(self.files[index])


class WrappedDataLoader:
    """
    a useful helper for wrapping data loaders with functions we might want to apply
    without having to hardcode it into our Custom dataset
    """

    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield self.func(*b)


def mount_to_device(x, y, dev):
    # mount x and y to a given device
    return x.to(dev), y.to(dev)


def get_jpgs_from_path(path):
    """
    recursively list through a given directory and get all the jpg file paths
    :param path: the starting file
    :return: list of all jpgs underneath the path
    """
    imgs = []
    for _ in os.listdir(path):
        p = f"{path}/{_}"
        if os.path.isdir(p):
            dir_ims = get_jpgs_from_path(p)
            imgs = imgs + dir_ims
        if ".jpg" in p:
            imgs.append(p)
    return imgs


def sample_from_data_loader(data_loader):
    # sample one batch from a dataloader
    return next(iter(data_loader))


def get_test_64batch_from_path(path, wrapped=None, dev="cpu", seed=42):
    data = Data(path,
                augmented=False,
                workers=0,
                device=dev,
                test_amt=1000,
                batch_size=64,
                wrapped_function=wrapped,
                seed=seed)
    x, y = sample_from_data_loader(data.get_test_data())
    return x, y

