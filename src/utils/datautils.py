import torch
import torchvision
import numpy as np
import random
import os

from torch.utils.data import Dataset


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CustomDataset(Dataset):

    def __init__(self, files):
        self.files = files

    def preprocess(self, file):
        y = int(file[-5:-4])
        img = torchvision.io.read_image(file)
        x = img.float()
        return x, y

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.preprocess(self.files[index])


class WrappedDataLoader:
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
    return x.to(dev), y.to(dev)


def get_jpgs_from_path(path):
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
    return next(iter(data_loader))
